from __future__ import annotations

import csv
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metrics import (
    diagnosis_samples,
    fabrication_rate,
    jsd,
    load_records,
    samples_to_distribution,
)
from normalize import extract_diagnosis
from utils import DATA_AGG, DATA_CAT, load_taxonomy

MODELS = ["claude-opus-4-7", "gpt-5.4", "gemini-3.1-pro-preview"]
DOMAINS = ["xray", "mri", "derm"]
FACTORIAL = [
    "32wm", "32wf", "32bm", "32bf", "32rm", "32rf",
    "65wm", "65wf", "65bm", "65bf", "65rm", "65rf",
]
FIELDS = ["m", "d", "cell", "fab0", "fab1", "d_fab", "overall", "wf", "n0", "n1"]


def main() -> int:
    rows = []
    for model, domain in product(MODELS, DOMAINS):
        taxonomy = load_taxonomy(domain)["categories"]
        cats = list(dict.fromkeys(list(taxonomy) + ["Other", "NO_DIAGNOSIS"]))
        base = DATA_CAT / f"E1_{model}_{domain}_D0.jsonl"
        if not base.exists():
            continue
        r0 = load_records(base)
        s0_all = [extract_diagnosis(r, domain) for r in r0]
        s0_fab = diagnosis_samples(r0, domain, mirage_only=True)
        fab0 = fabrication_rate(r0)
        for cell in FACTORIAL:
            p = DATA_CAT / f"E1_{model}_{domain}_{cell}.jsonl"
            if not p.exists():
                continue
            r1 = load_records(p)
            s1_all = [extract_diagnosis(r, domain) for r in r1]
            s1_fab = diagnosis_samples(r1, domain, mirage_only=True)
            fab1 = fabrication_rate(r1)
            overall = jsd(
                samples_to_distribution(s0_all or ["NO_DIAGNOSIS"], cats),
                samples_to_distribution(s1_all or ["NO_DIAGNOSIS"], cats),
            )
            if s0_fab and s1_fab:
                wf = jsd(
                    samples_to_distribution(s0_fab, cats),
                    samples_to_distribution(s1_fab, cats),
                )
            else:
                wf = ""
            rows.append([model, domain, cell, fab0, fab1, fab1 - fab0,
                         overall, wf, len(s0_fab), len(s1_fab)])
    out = DATA_AGG / "jsd_decomposition.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
