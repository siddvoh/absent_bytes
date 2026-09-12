"""E1: primary demographic-factorial experiment.

13 demographic conditions per (model, domain):
  D0 (neutral) + 2 ages x 2 sexes x 3 races = 1 + 12 = 13
3 models x 3 domains x 13 conditions x N=100 seeds = 11,700 calls.
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
from utils import LOGS, load_env, load_seeds

load_env()

DEMOGRAPHICS_ALL = [
    "D0",
    "32wm", "32wf", "32bm", "32bf", "32rm", "32rf",
    "65wm", "65wf", "65bm", "65bf", "65rm", "65rf",
]
DOMAINS_ALL = ["xray", "mri", "derm"]


def main(n: int, providers: list[str], domains: list[str], demographics: list[str]) -> None:
    seeds = load_seeds()[:n]
    assert len(seeds) == n, f"only {len(seeds)} seeds available, n={n}"

    t0 = time.time()
    summaries = []
    total_cells = len(providers) * len(domains) * len(demographics)
    cell_idx = 0
    for provider in providers:
        for domain in domains:
            for demog in demographics:
                cell_idx += 1
                print(f"\n[E1 {cell_idx}/{total_cells}] {provider} {domain} {demog} x {n}", flush=True)
                s = run_cell(provider, domain, demog, seeds, experiment="E1")
                summaries.append(s)
                spend = current_spend()
                print(f"  cell done: {s['count']} recs, {s['errors']} errors, "
                      f"{s['schema_failures']} schema_fail, {s['wall_s']:.1f}s")
                print(f"  running spend: anthropic=${spend.get('anthropic', 0):.2f} "
                      f"openai=${spend.get('openai', 0):.2f} "
                      f"vertex=${spend.get('vertex', 0):.2f} "
                      f"total=${spend.get('_total', 0):.2f}")

    elapsed = time.time() - t0
    print(f"\nE1 complete in {elapsed / 60:.1f} min across {cell_idx} cells")
    summary_path = LOGS / "run_summaries" / f"e1_{int(t0)}.json"
    with open(summary_path, "w") as f:
        json.dump({"elapsed_s": elapsed, "n_per_cell": n, "summaries": summaries}, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="samples per cell")
    p.add_argument("--providers", nargs="+", default=["anthropic", "openai", "gemini"])
    p.add_argument("--domains", nargs="+", default=DOMAINS_ALL)
    p.add_argument("--demographics", nargs="+", default=DEMOGRAPHICS_ALL)
    args = p.parse_args()
    main(args.n, args.providers, args.domains, args.demographics)
