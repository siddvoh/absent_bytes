from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metrics import jsd, load_records, samples_to_distribution
from normalize import extract_diagnosis
from utils import DATA_AGG, DATA_CAT, load_taxonomy

CELLS = [("claude-opus-4-7", "derm", "65wm"), ("gpt-5.4", "derm", "65rf")]
FIELDS = ["model", "domain", "demographic", "probe", "jsd_vs_D0", "n"]


def samples(path: Path, domain: str) -> list[str]:
    return [extract_diagnosis(r, domain) for r in load_records(path)]


def main() -> int:
    rows = []
    for model, domain, cell in CELLS:
        taxonomy = load_taxonomy(domain)["categories"]
        cats = list(dict.fromkeys(list(taxonomy) + ["Other", "NO_DIAGNOSIS"]))
        base = DATA_CAT / f"E1_{model}_{domain}_D0.jsonl"
        if not base.exists():
            continue
        d0 = samples_to_distribution(samples(base, domain) or ["NO_DIAGNOSIS"], cats)
        for probe, path in (
            ("mole", DATA_CAT / f"E1_{model}_{domain}_{cell}.jsonl"),
            ("lesion", DATA_CAT / f"E4_{model}_{domain}_{cell}_lesion.jsonl"),
        ):
            if not path.exists():
                continue
            s = samples(path, domain)
            rows.append([model, domain, cell, probe,
                         jsd(d0, samples_to_distribution(s or ["NO_DIAGNOSIS"], cats)), len(s)])
    out = DATA_AGG / "e4_probe_noun.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        w.writerows(rows)
    for r in rows:
        print(f"{r[0]}\t{r[2]}\t{r[3]}\tJSD={r[4]:.4f}\tn={r[5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
