"""Mirage-detection judge (Asadi §7.3 Phantom-0, verbatim).

Runs GPT-5.4-nano via the OpenAI Responses API over every record's
`reasoning + key_findings` fields. Sets the `is_mirage` boolean as a
belt-and-suspenders cross-check against the model's self-reported
`image_present` field.

No diagnosis categorization judge. Taxonomy mapping is deterministic
(src.normalize.extract_diagnosis).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import time
from dataclasses import dataclass

from openai import OpenAI, APIError

from utils import load_judge_config, load_models_config
from cost_tracker import estimate_cost

_client: OpenAI | None = None


def _oai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


@dataclass
class MirageJudgeOutput:
    is_mirage: bool | None    # True = mirage; False = acknowledged missing image;
    raw_judge_text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    parse_ok: bool = True
    error: str | None = None


_ANS = re.compile(r"<answer>\s*(true|false)\s*</answer>", re.IGNORECASE)


def _judge_call(prompt: str, cfg: dict) -> tuple[str, int, int]:
    """Call the judge model via Responses API.

    Per Asadi §7.3, ONLY model + prompt + output format are specified by the
    protocol. Every other parameter is left at the API default so our
    invocation literally matches theirs (up to the version upgrade to nano).
    The one non-protocol parameter we set is max_output_tokens — a
    runaway-cost safety cap.
    """
    kwargs = {
        "model": cfg["model"],
        "input": prompt,
        "max_output_tokens": cfg.get("max_output_tokens", 4000),
    }
    resp = _oai().responses.create(**kwargs)
    text = resp.output_text or ""
    usage = resp.usage
    in_tok = usage.input_tokens if usage else 0
    out_tok = usage.output_tokens if usage else 0
    return text, in_tok, out_tok


def judge_record_text(text: str) -> MirageJudgeOutput:
    cfg = load_models_config()["judge"]
    prompt = load_judge_config()["mirage_detection"].format(model_response=text)
    try:
        raw, in_tok, out_tok = _judge_call(prompt, cfg)
    except APIError as e:
        return MirageJudgeOutput(None, "", 0, 0, 0.0, parse_ok=False, error=f"judge:{e}")
    m = _ANS.search(raw)
    if not m:
        return MirageJudgeOutput(
            is_mirage=None,
            raw_judge_text=raw,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost(in_tok, out_tok, cfg),
            parse_ok=False,
        )
    # Asadi §7.3 convention:
    #   <answer>true</answer>  = response mentions missing image -> NOT a mirage
    #   <answer>false</answer> = response does NOT mention it    -> IS a mirage
    is_mirage = m.group(1).lower() == "false"
    return MirageJudgeOutput(
        is_mirage=is_mirage,
        raw_judge_text=raw,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=estimate_cost(in_tok, out_tok, cfg),
        parse_ok=True,
    )


def _extract_judgeable_text(record: dict) -> str:
    """Combine reasoning + key_findings for judging (both are schema-required)."""
    payload = record.get("response_json") or {}
    parts = []
    r = payload.get("reasoning")
    if r:
        parts.append(r)
    k = payload.get("key_findings")
    if k:
        parts.append(k)
    if not parts:
        return record.get("response_text", "") or ""
    return "\n\n".join(parts)


def judge_file(raw_jsonl_path: str, out_path: str, max_workers: int = 20) -> dict:
    """Run mirage-detection judge on every record. Preserves input order.

    Adds two fields to each record:
      is_mirage (bool)        — Asadi §7.3 judge verdict
      judge_mirage_raw (str)  — raw judge response text
    """
    t0 = time.time()
    with open(raw_jsonl_path) as f:
        records = [json.loads(line) for line in f]

    total_cost = 0.0
    errors = 0

    def _judge_one(idx_rec):
        i, rec = idx_rec
        if rec.get("error"):
            rec["is_mirage"] = None
            rec["judge_parse_ok"] = False
            rec["judge_mirage_raw"] = ""
            rec["judge_cost_usd"] = 0.0
            return i, rec, 0.0, True
        text = _extract_judgeable_text(rec)
        j = judge_record_text(text)
        rec["is_mirage"] = j.is_mirage
        rec["judge_parse_ok"] = j.parse_ok
        rec["judge_mirage_raw"] = j.raw_judge_text
        rec["judge_cost_usd"] = j.cost_usd
        rec["judge_error"] = j.error
        return i, rec, j.cost_usd, bool(j.error) or not j.parse_ok

    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_judge_one, (i, r)) for i, r in enumerate(records)]
        done = 0
        for fut in cf.as_completed(futures):
            i, rec, cost, was_err = fut.result()
            records[i] = rec
            if was_err:
                errors += 1
            total_cost += cost
            done += 1
            if done % 100 == 0:
                print(f"  judged {done}/{len(records)} / cost ${total_cost:.4f}", flush=True)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "input": str(raw_jsonl_path),
        "output": str(out_path),
        "n_processed": len(records),
        "errors": errors,
        "total_cost_usd": total_cost,
        "wall_s": time.time() - t0,
    }
