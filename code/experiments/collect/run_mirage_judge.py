"""Run the Asadi §7.3 mirage-detection judge on every record.

Belt-and-suspenders cross-check against the model's self-reported
`image_present` boolean. Reads each JSONL under data/raw/, writes a copy
to data/categorized/ with two additional fields: `is_mirage` (bool) and
`judge_mirage_raw` (str).

No diagnosis-categorization pass. Taxonomy mapping is deterministic and
happens in src/analyze/run_analysis.py via src.normalize.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from judging import judge_file
from utils import DATA_CAT, DATA_RAW, LOGS, load_env

load_env()


def main(filters: list[str] | None) -> None:
    files = sorted(DATA_RAW.glob("*.jsonl"))
    if filters:
        files = [f for f in files if any(tok in f.name for tok in filters)]
    if not files:
        print(f"No files in {DATA_RAW} matching filters={filters}")
        return

    DATA_CAT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summaries = []
    for fp in files:
        out_fp = DATA_CAT / fp.name
        print(f"[judge] {fp.name}", flush=True)
        s = judge_file(str(fp), str(out_fp))
        summaries.append(s)
        print(f"  done: {s['n_processed']} records, {s['errors']} errors, "
              f"cost=${s['total_cost_usd']:.4f}, wall={s['wall_s']:.1f}s")

    elapsed = time.time() - t0
    total_cost = sum(s["total_cost_usd"] for s in summaries)
    print(f"\nAll files done in {elapsed / 60:.1f} min, total judge cost=${total_cost:.4f}")
    summary_path = LOGS / "run_summaries" / f"judge_{int(t0)}.json"
    with open(summary_path, "w") as f:
        json.dump({"elapsed_s": elapsed, "total_cost_usd": total_cost, "files": summaries}, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--filter", nargs="*", default=None,
                   help="optional substrings to filter filenames (e.g. E1 claude)")
    args = p.parse_args()
    main(args.filter)
