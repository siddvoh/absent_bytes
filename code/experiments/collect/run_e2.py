"""E2: paraphrase ablation.

Claude Opus 4.7, chest X ray D0 prompt paraphrased 3 ways x N seeds.
Establishes the paraphrase noise floor for JSD claim comparisons.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cost_tracker import current_spend
from sampling import run_cell
from utils import LOGS, load_env, load_prompts_config, load_seeds

load_env()


def main(n: int) -> None:
    seeds = load_seeds()[:n]
    e2_prompts = load_prompts_config()["e2"]
    t0 = time.time()
    summaries = []
    for demog_key in ("D0_P1", "D0_P2", "D0_P3"):
        cell_key = f"xray_{demog_key}"
        print(f"\n[E2] anthropic xray {demog_key} x {n}", flush=True)
        s = run_cell(
            provider="anthropic",
            domain="xray",
            demographic=demog_key,
            seeds=seeds,
            experiment="E2",
            prompts_override={cell_key: e2_prompts[cell_key]},
        )
        summaries.append(s)
        spend = current_spend()
        print(f"  cell done: {s['count']} recs, {s['errors']} errors, "
              f"{s['schema_failures']} schema_fail, {s['wall_s']:.1f}s")
        print(f"  anthropic spend: ${spend.get('anthropic', 0):.2f}")

    elapsed = time.time() - t0
    print(f"\nE2 complete in {elapsed / 60:.1f} min")
    summary_path = LOGS / "run_summaries" / f"e2_{int(t0)}.json"
    with open(summary_path, "w") as f:
        json.dump({"elapsed_s": elapsed, "n_per_cell": n, "summaries": summaries}, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()
    main(args.n)
