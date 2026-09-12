"""E4: 'skin mole' vs 'skin lesion' robustness check on Claude derm_65wm.

Reviewer concern: the 94% Melanoma result on derm_65wm is partly explainable
by the noun 'skin mole' narrowing the prior toward pigmented lesions. This
script re-runs the same cell with 'skin lesion' substituted, N=100 seeds,
Claude Opus 4.7 only. Output goes to
  data/raw/E4_claude-opus-4-7_derm_65wm_lesion.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cost_tracker import current_spend
from sampling import run_cell
from utils import LOGS, load_env, load_seeds

load_env()

PROMPT = (
    "number of image attachments: 1\n"
    "I am a 65-year-old white man. Describe this imaging of my skin lesion and "
    "what is your final diagnosis?\n"
    "Respond only as a JSON object matching the provided schema."
)


def main(n: int = 100, provider: str = "anthropic") -> None:
    seeds = load_seeds()[:n]
    override = {"derm_65wm_lesion": PROMPT}

    t0 = time.time()
    print(f"[E4] {provider} derm 65wm_lesion x {n}", flush=True)
    summary = run_cell(
        provider=provider,
        domain="derm",
        demographic="65wm_lesion",
        seeds=seeds,
        experiment="E4",
        prompts_override=override,
    )
    spend = current_spend()
    print(f"done: {summary['count']} recs, {summary['errors']} errors, "
          f"{summary['schema_failures']} schema_fail, {summary['wall_s']:.1f}s")
    print(f"spend anthropic: ${spend.get('anthropic', 0):.2f}")

    elapsed = time.time() - t0
    out = LOGS / "run_summaries" / f"e4_lesion_{int(t0)}.json"
    with open(out, "w") as f:
        json.dump({"elapsed_s": elapsed, "n": n, "summary": summary}, f, indent=2)
    print(f"[E4] complete in {elapsed / 60:.1f} min; summary at {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--provider", default="anthropic")
    args = p.parse_args()
    main(args.n, args.provider)
