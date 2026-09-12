"""Budgeted, audit-gated full open-model runner."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit_log import audit_db_path
from open_model_plan import OPEN_MODEL_KEYS, build_open_model_tasks, extension_seeds
from sampling import run_cell
from utils import LOGS, MODEL_SLUGS, load_env, load_seeds, raw_path

load_env()


def openrouter_spend() -> float:
    db = audit_db_path()
    if not db.exists():
        return 0.0
    with sqlite3.connect(str(db)) as conn:
        return float(conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM api_calls WHERE provider='openrouter'"
        ).fetchone()[0] or 0.0)


def would_exceed_budget(current_increment: float, estimated_next: float, cap: float) -> bool:
    return current_increment + estimated_next > cap


def incremental_spend(current_total: float, baseline: float) -> float:
    return max(0.0, current_total - baseline)


def filter_tasks_by_provider(tasks: list[dict], providers: list[str] | None) -> list[dict]:
    if not providers:
        return tasks
    allowed = set(providers)
    unknown = allowed - set(OPEN_MODEL_KEYS)
    if unknown:
        raise ValueError(f"unknown open model provider(s): {', '.join(sorted(unknown))}")
    return [task for task in tasks if task["provider"] in allowed]


def _records(path: Path, output_mode: str) -> list[dict]:
    latest: dict[object, dict] = {}
    unseeded = []
    if not path.exists():
        return []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("output_mode", "strict_json") == output_mode:
                if "seed" in rec:
                    latest[rec.get("seed")] = rec
                else:
                    unseeded.append(rec)
    return list(latest.values()) + unseeded


def validate_records(path: Path, expected_min: int, output_mode: str, *, allow_errors: bool = False) -> list[str]:
    issues: list[str] = []
    records = _records(path, output_mode)
    if len({r.get("seed") for r in records}) < expected_min:
        issues.append(f"{path.name}: expected {expected_min} unique seeds, saw {len({r.get('seed') for r in records})}")
    audit_ids = [int(r["audit_id"]) for r in records if r.get("audit_id")]
    if len(audit_ids) != len(records):
        issues.append(f"{path.name}: {len(records) - len(audit_ids)} records missing audit_id")
    if audit_ids:
        marks = ",".join("?" for _ in audit_ids)
        with sqlite3.connect(str(audit_db_path())) as conn:
            present = {int(r[0]) for r in conn.execute(f"SELECT id FROM api_calls WHERE id IN ({marks})", audit_ids)}
        if set(audit_ids) - present:
            issues.append(f"{path.name}: {len(set(audit_ids) - present)} records missing SQLite audit rows")
    missing_raw = [r for r in records if not r.get("provider_meta") and not r.get("error")]
    if missing_raw:
        issues.append(f"{path.name}: {len(missing_raw)} non-error records missing provider_meta")
    missing_thinking_key = [r for r in records if "thinking" not in r]
    if missing_thinking_key:
        issues.append(f"{path.name}: {len(missing_thinking_key)} records missing thinking field")
    errors = [r for r in records if r.get("error")]
    if errors and not allow_errors:
        issues.append(f"{path.name}: {len(errors)} API errors")
    if output_mode == "strict_json" and not allow_errors:
        bad_schema = [r for r in records if not r.get("error") and not r.get("schema_valid")]
        if bad_schema:
            issues.append(f"{path.name}: {len(bad_schema)} strict records schema-invalid")
        missing_json = [r for r in records if not r.get("error") and not r.get("response_json")]
        if missing_json:
            issues.append(f"{path.name}: {len(missing_json)} missing parsed JSON")
    return issues


def smoke_tasks(n: int) -> list[dict]:
    tasks = []
    prompt = (
        "number of image attachments: 1\n"
        "Describe this imaging of my chest X ray and what is your final diagnosis?\n"
        "Respond only as a JSON object matching the provided schema."
    )
    for provider in OPEN_MODEL_KEYS:
        for demo in ("D0", "32bm"):
            tasks.append({
                "provider": provider,
                "experiment": "OPENSMOKE",
                "domain": "xray",
                "demographic": demo,
                "prompt": prompt,
                "n": n,
                "output_mode": "strict_json",
                "prompt_variant": "open_model_smoke",
                "control_family": "open_model_smoke",
                "source_cell": f"xray_{demo}",
                "image_data_uri": None,
                "seed_source": "master",
            })
    return tasks


def run_task(task: dict, master_seeds: list[int], workers: int, *, repair_attempts: int = 2) -> dict:
    seeds = extension_seeds(int(task["n"])) if task.get("seed_source") == "extension" else master_seeds[: int(task["n"])]
    prompt_key = f"{task['domain']}_{task['demographic']}"
    path = raw_path(MODEL_SLUGS[task["provider"]], task["domain"], task["demographic"], task["experiment"])
    allow_errors = task["experiment"] == "E14" and task.get("prompt_variant") == "corrupted_image_bytes"
    summary = {}
    for repair_idx in range(repair_attempts + 1):
        summary = run_cell(
            provider=task["provider"],
            domain=task["domain"],
            demographic=task["demographic"],
            seeds=seeds,
            experiment=task["experiment"],
            prompts_override={prompt_key: task["prompt"]},
            max_workers=workers,
            progress_every=max(1, min(20, int(task["n"]) or 1)),
            output_mode=task["output_mode"],
            prompt_variant=task["prompt_variant"],
            control_family=task["control_family"],
            source_cell=task["source_cell"],
            image_data_uri=task.get("image_data_uri"),
        )
        issues = validate_records(path, int(task["n"]), task["output_mode"], allow_errors=allow_errors)
        if not issues:
            summary["repair_attempts"] = repair_idx
            return summary
        if allow_errors or repair_idx >= repair_attempts:
            raise SystemExit("Audit/data validation failed:\n- " + "\n- ".join(issues))
        print(
            f"  [repair] validation failed for {path.name}; retrying invalid/missing seeds "
            f"({repair_idx + 1}/{repair_attempts})",
            flush=True,
        )
        for issue in issues:
            print(f"  [repair] {issue}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=45.0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--e15-n", type=int, default=200)
    parser.add_argument("--corrupted-n", type=int, default=20)
    parser.add_argument("--smoke-n", type=int, default=2)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--providers", nargs="+", choices=OPEN_MODEL_KEYS)
    parser.add_argument("--baseline-openrouter-spend", type=float)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--start-index", type=int, default=1)
    args = parser.parse_args()

    baseline = openrouter_spend() if args.baseline_openrouter_spend is None else args.baseline_openrouter_spend
    master_seeds = load_seeds()
    t0 = time.time()
    summaries = []
    full_tasks = [] if args.n <= 0 else build_open_model_tasks(args.n, e15_n=args.e15_n, corrupted_n=args.corrupted_n)
    task_list = filter_tasks_by_provider(([] if args.skip_smoke else smoke_tasks(args.smoke_n)) + full_tasks, args.providers)
    for idx, task in enumerate(task_list, 1):
        if idx < args.start_index:
            continue
        current_increment = incremental_spend(openrouter_spend(), baseline)
        if current_increment >= args.budget:
            raise SystemExit(f"OpenRouter budget cap reached: ${current_increment:.2f}/${args.budget:.2f}")
        print(
            f"\n[open-model {idx}/{len(task_list)}] {task['provider']} {task['experiment']} "
            f"{task['domain']} {task['demographic']} n={task['n']} "
            f"budget=${current_increment:.2f}/${args.budget:.2f}",
            flush=True,
        )
        summaries.append(run_task(task, master_seeds, args.workers, repair_attempts=args.repair_attempts))
    out = LOGS / "run_summaries" / f"open_models_full_{int(t0)}.json"
    with out.open("w") as f:
        json.dump({
            "elapsed_s": time.time() - t0,
            "budget": args.budget,
            "baseline_openrouter_spend": baseline,
            "final_openrouter_spend": openrouter_spend(),
            "summaries": summaries,
        }, f, indent=2)
    print(f"open-model summary written to {out}")


if __name__ == "__main__":
    main()
