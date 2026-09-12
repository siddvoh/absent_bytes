"""Concurrent sampling loop (v2).

Persists one JSONL record per call containing:
  - full parsed JSON payload (response_json)
  - raw JSON string (response_text)
  - full CoT / reasoning summary (thinking)    — where provider exposes it
  - provider metadata (stop_reason, thought token counts, model version)
  - input/output token counts, latency, cost

Resume semantics: if a cell's output file already contains a non-error
record for a given seed, that seed is skipped.
"""
from __future__ import annotations

import concurrent.futures as cf
import json as _json
import time
from typing import Iterable

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from clients.anthropic_client import AnthropicClient
from clients.gemini_client import GeminiVertexClient
from clients.openai_client import OpenAIClient
from clients.base import BaseClient, CallResult
from cost_tracker import log_cost
from utils import (
    CallRecord,
    MODEL_SLUGS,
    append_jsonl,
    error_log_path,
    load_models_config,
    load_prompts_config,
    load_seeds,
    now_iso,
    raw_path,
)


def get_client(provider: str) -> BaseClient:
    cfg = load_models_config()[provider]
    if provider == "anthropic":
        return AnthropicClient(cfg)
    if provider == "openai":
        return OpenAIClient(cfg)
    if provider == "gemini":
        return GeminiVertexClient(cfg)
    raise ValueError(f"unknown provider: {provider}")


_TRANSIENT_ERROR_TOKENS = ("rate", "timeout", "overloaded", "503", "502", "429", "resource_exhausted")


@retry(stop=stop_after_attempt(4), wait=wait_exponential_jitter(initial=2, max=30))
def _call_with_retry(client: BaseClient, prompt: str, seed: int) -> CallResult:
    result = client.call(prompt, seed=seed)
    if result.error and any(tok in result.error.lower() for tok in _TRANSIENT_ERROR_TOKENS):
        raise RuntimeError(result.error)
    return result


def _load_existing_seeds(out_path) -> set[int]:
    existing: set[int] = set()
    if not out_path.exists():
        return existing
    with open(out_path) as f:
        for line in f:
            try:
                r = _json.loads(line)
                if not r.get("error"):
                    existing.add(int(r["seed"]))
            except Exception:
                pass
    return existing


def run_cell(
    provider: str,
    domain: str,
    demographic: str,
    seeds: list[int],
    experiment: str = "E1",
    prompts_override: dict | None = None,
    max_workers: int | None = None,
    progress_every: int = 10,
) -> dict:
    """Run all `seeds` for one (provider, domain, demographic) cell.

    `prompts_override` — if given, supersedes the prompts.yaml lookup.
    Must contain the `{domain}_{demographic}` key.
    """
    cfg_all = load_models_config()
    cfg = cfg_all[provider]
    exp_lower = experiment.lower()
    prompt_section = exp_lower if exp_lower in ("e1", "e2", "e2b") else "e1"
    prompts = prompts_override or load_prompts_config()[prompt_section]
    prompt_key = f"{domain}_{demographic}"
    if prompt_key not in prompts:
        raise KeyError(f"prompt key {prompt_key} not in prompts config section {prompt_section}")
    prompt = prompts[prompt_key]

    max_workers = max_workers or cfg.get("max_concurrent", 3)
    client = get_client(provider)
    model_slug = MODEL_SLUGS[provider]
    out_path = raw_path(model_slug, domain, demographic, experiment)

    existing_seeds = _load_existing_seeds(out_path)
    if existing_seeds:
        print(f"  [resume] {len(existing_seeds)} seeds already succeeded in {out_path.name}", flush=True)
    pending = [s for s in seeds if s not in existing_seeds]

    t0 = time.time()
    completed = 0
    errors = 0
    schema_failures = 0

    def _one(seed: int) -> tuple[int, CallResult]:
        try:
            return seed, _call_with_retry(client, prompt, seed)
        except Exception as e:
            return seed, CallResult(error=str(e))

    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, s) for s in pending]
        for fut in cf.as_completed(futures):
            seed, result = fut.result()
            cost = client.estimate_cost(result.input_tokens, result.output_tokens)
            record = CallRecord(
                timestamp=now_iso(),
                model=cfg["model"],
                provider=provider if provider != "gemini" else "vertex",
                domain=domain,
                demographic=demographic,
                seed=seed,
                experiment=experiment,
                prompt=prompt,
                response_text=result.text,
                response_json=result.parsed,
                schema_valid=result.schema_valid,
                thinking=result.thinking,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=cost,
                latency_ms=result.latency_ms,
                provider_meta=result.raw,
                error=result.error,
            )
            append_jsonl(out_path, record)
            log_cost(record)
            if result.error:
                errors += 1
                append_jsonl(error_log_path(), {
                    "timestamp": record.timestamp,
                    "provider": provider,
                    "model": cfg["model"],
                    "domain": domain,
                    "demographic": demographic,
                    "experiment": experiment,
                    "seed": seed,
                    "error": result.error,
                })
            elif not result.schema_valid:
                schema_failures += 1
            completed += 1
            if completed % progress_every == 0:
                dt = time.time() - t0
                rps = completed / max(dt, 1e-6)
                print(f"  [{provider}:{domain}:{demographic}] {completed}/{len(pending)} "
                      f"(errors={errors}, schema_fail={schema_failures}, {rps:.2f}/s, {dt:.1f}s)",
                      flush=True)

    return {
        "provider": provider,
        "domain": domain,
        "demographic": demographic,
        "experiment": experiment,
        "count": completed,
        "errors": errors,
        "schema_failures": schema_failures,
        "path": str(out_path),
        "wall_s": time.time() - t0,
    }


def run_many(
    providers: Iterable[str],
    domains: Iterable[str],
    demographics: Iterable[str],
    seeds_per_cell: dict[str, int] | int,
    experiment: str = "E1",
) -> list[dict]:
    master_seeds = load_seeds()
    summaries = []
    for provider in providers:
        n = seeds_per_cell if isinstance(seeds_per_cell, int) else seeds_per_cell[provider]
        seeds = master_seeds[:n]
        for domain in domains:
            for demographic in demographics:
                print(f"[run] {provider} {domain} {demographic} × {n}", flush=True)
                summary = run_cell(provider, domain, demographic, seeds, experiment=experiment)
                summaries.append(summary)
    return summaries
