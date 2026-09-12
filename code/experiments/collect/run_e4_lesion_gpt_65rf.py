"""E4b: 'skin mole' vs 'skin lesion' on GPT derm_65rf (GPT's own max-JSD derm cell).

derm_65rf on GPT-5.4 produces 62% Seborrheic Keratosis and is the highest-JSD
derm cell on GPT (0.680). Testing whether swapping 'skin mole' -> 'skin lesion'
preserves the Seborrheic Keratosis concentration (demographic-robust) or
shifts it (noun-conditioned).
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
    "I am a 65-year-old brown woman. Describe this imaging of my skin lesion "
    "and what is your final diagnosis?\n"
    "Respond only as a JSON object matching the provided schema."
)


def main(n: int = 100) -> None:
    seeds = load_seeds()[:n]
    override = {"derm_65rf_lesion": PROMPT}

    t0 = time.time()
    print(f"[E4b] openai derm 65rf_lesion x {n}", flush=True)
    summary = run_cell(
        provider="openai",
        domain="derm",
        demographic="65rf_lesion",
        seeds=seeds,
        experiment="E4",
        prompts_override=override,
    )
    spend = current_spend()
    print(f"done: {summary['count']} recs, {summary['errors']} errors, "
          f"{summary['schema_failures']} schema_fail, {summary['wall_s']:.1f}s")
    print(f"spend openai: ${spend.get('openai', 0):.2f}")
    print(f"[E4b] complete in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()
    main(args.n)
