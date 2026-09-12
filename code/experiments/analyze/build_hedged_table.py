from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metrics import _has_diagnosis, load_records
from utils import DATA_AGG, DATA_CAT

CELLS = [
    ("claude-opus-4-7", "derm", "65wm"),
    ("gpt-5.4", "xray", "32bm"),
    ("gemini-3.1-pro-preview", "xray", "32bf"),
]
FIELDS = ["model", "domain", "demographic", "judge_ack_missing", "diagnosis_filled", "count"]


def main() -> int:
    rows = []
    for model, domain, cell in CELLS:
        path = DATA_CAT / f"E1_{model}_{domain}_{cell}.jsonl"
        if not path.exists():
            continue
        counts = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
        for record in load_records(path):
            counts[(record.get("is_mirage") is False, _has_diagnosis(record))] += 1
        for (ack, filled), n in sorted(counts.items(), reverse=True):
            rows.append([model, domain, cell, ack, filled, n])
    out = DATA_AGG / "hedged_table.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        w.writerows(rows)
    for r in rows:
        print(*r, sep="\t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
