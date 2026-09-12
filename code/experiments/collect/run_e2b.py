"""E2b: paraphrase ablation on each provider's top-fabrication cell.

Exploratory follow-up to pre-registered E2 (DEV-015). Each provider's
top-fab cell gets 3 paraphrases × N=100 seeds, same structure as E2.

Top cells (from N=100 E1 data):
  anthropic: derm_65wm   (94% Melanoma)
  openai:    xray_32bm   (77% Sarcoidosis)
  gemini:    not selected yet (fabrication rate too low)
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

TOP_CELL = {
    "anthropic": {"domain": "derm", "prefix": "65wm"},
    "openai":    {"domain": "xray", "prefix": "32bm"},
    "gemini":    {"domain": "xray", "prefix": "32bf"},
}
PARAPHRASE_SUFFIXES = ["P1", "P2", "P3"]


def main(n: int, providers: list[str]) -> None:
    seeds = load_seeds()[:n]
    prompts_e2b = load_prompts_config()["e2b"]

    t0 = time.time()
    summaries = []
    for provider in providers:
        if provider not in TOP_CELL:
            print(f"[E2b] skipping {provider}: no top cell configured", flush=True)
            continue
        spec = TOP_CELL[provider]
        domain = spec["domain"]
        prefix = spec["prefix"]
        for p_suffix in PARAPHRASE_SUFFIXES:
            demographic = f"{prefix}_{p_suffix}"
            prompt_key = f"{domain}_{demographic}"
            prompt_text = prompts_e2b[prompt_key]
            print(f"\n[E2b] {provider} {domain} {demographic} x {n}", flush=True)
            s = run_cell(
                provider=provider,
                domain=domain,
                demographic=demographic,
                seeds=seeds,
                experiment="E2b",
                prompts_override={prompt_key: prompt_text},
            )
            summaries.append(s)
            spend = current_spend()
            print(f"  cell done: {s['count']} recs, {s['errors']} errors, "
                  f"{s['schema_failures']} schema_fail, {s['wall_s']:.1f}s")
            print(f"  spend: anthropic=${spend.get('anthropic', 0):.2f} "
                  f"openai=${spend.get('openai', 0):.2f} "
                  f"total=${spend.get('_total', 0):.2f}")

    elapsed = time.time() - t0
    print(f"\nE2b complete in {elapsed / 60:.1f} min")
    path = LOGS / "run_summaries" / f"e2b_{int(t0)}.json"
    with open(path, "w") as f:
        json.dump({"elapsed_s": elapsed, "n_per_cell": n, "summaries": summaries}, f, indent=2)
    print(f"Summary: {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--providers", nargs="+", default=["anthropic", "openai", "gemini"])
    args = p.parse_args()
    main(args.n, args.providers)
