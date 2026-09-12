from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from normalize import extract_diagnosis
from utils import REPO_ROOT

RECORDS = REPO_ROOT / "data" / "records_E1"
OUT = REPO_ROOT / "data" / "aggregated" / "diagnosis_distribution_counts.csv"
FIELDS = ["model", "domain", "demographic", "category", "count", "n_samples", "source_sha256"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_key(path: Path) -> tuple[str, str, str]:
    stem = path.stem.split("_", 1)[1]
    model, domain, demographic = stem.rsplit("_", 2)
    return model, domain, demographic


def count_cell(path: Path, domain: str) -> tuple[Counter, int]:
    seen: dict[int, str] = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        seed = record.get("seed")
        if seed in seen:
            continue
        seen[seed] = extract_diagnosis(record, domain)
    return Counter(seen.values()), len(seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=RECORDS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = []
    for path in sorted(args.records.glob("E1_*.jsonl")):
        model, domain, demographic = cell_key(path)
        counts, n = count_cell(path, domain)
        digest = sha256(path)
        for category, count in sorted(counts.items()):
            rows.append([model, domain, demographic, category, count, n, digest])

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
