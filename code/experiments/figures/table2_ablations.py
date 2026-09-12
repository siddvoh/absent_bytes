"""Table 2: demographic JSD vs paraphrase + seed noise floors.

Computes:
  - Median demographic JSD per provider (out of 36 cells each)
  - E2 paraphrase JSD (Claude D0 × 3 paraphrases) — degenerate, for completeness
  - E2b paraphrase JSD on each provider's top-fab cell (non-degenerate floor)
  - E3 seed noise floor (Claude xray D0 three-way split)
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from utils import DATA_AGG, DATA_CAT, REPO_ROOT, load_taxonomy
from metrics import bootstrap_jsd, load_records
from normalize import extract_diagnosis


PROVIDERS = ["claude-opus-4-7", "gpt-5.4", "gemini-3.1-pro-preview"]
MODEL_LABEL = {
    "claude-opus-4-7": "Claude Opus 4.7",
    "gpt-5.4": "GPT-5.4",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}
E2B_TOP = {
    "claude-opus-4-7": ("derm", "65wm"),
    "gpt-5.4": ("xray", "32bm"),
    "gemini-3.1-pro-preview": ("xray", "32bf"),
}


def _samples_from(path, domain):
    recs = load_records(path)
    if not recs: return None
    return [extract_diagnosis(r, domain) for r in recs]


def e2b_pairwise_jsd(provider):
    """Return the median of the three pairwise JSDs between canonical + P1 + P2 + P3."""
    domain, prefix = E2B_TOP[provider]
    tax = load_taxonomy(domain)["categories"] + ["Other", "NO_DIAGNOSIS"]

    variants = {
        "canonical": DATA_CAT / f"E1_{provider}_{domain}_{prefix}.jsonl",
        "P1":        DATA_CAT / f"E2b_{provider}_{domain}_{prefix}_P1.jsonl",
        "P2":        DATA_CAT / f"E2b_{provider}_{domain}_{prefix}_P2.jsonl",
        "P3":        DATA_CAT / f"E2b_{provider}_{domain}_{prefix}_P3.jsonl",
    }
    samples = {k: _samples_from(p, domain) for k, p in variants.items()}
    if any(v is None for v in samples.values()):
        return None

    pairs = list(combinations(["P1", "P2", "P3"], 2))
    jsds = []
    for a, b in pairs:
        bs = bootstrap_jsd(samples[a], samples[b], tax, n_bootstrap=500)
        jsds.append(bs["point"])
    return {
        "domain": domain, "cell": prefix,
        "jsds_P1P2_P1P3_P2P3": jsds,
        "max_jsd": max(jsds), "median_jsd": float(np.median(jsds)),
    }


def main():
    jsd_df = pd.read_parquet(DATA_AGG / "jsd_matrix.parquet")
    out_dir = REPO_ROOT / "data" / "derived" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in PROVIDERS:
        sub = jsd_df[jsd_df.model == p]
        rows.append([f"{MODEL_LABEL[p]} demographic JSD (36 cells)",
                     f"{sub.jsd_point.median():.3f}", f"{sub.jsd_point.max():.3f}", ""])

    e2_csv = DATA_AGG / "paraphrase_jsd.csv"
    if e2_csv.exists() and e2_csv.stat().st_size > 0:
        try:
            e2 = pd.read_csv(e2_csv)
            e2_max = e2.jsd_point.max() if not e2.empty else 0
            rows.append(["E2 paraphrase floor (Claude xray D0, pre-reg)",
                         "", f"{e2_max:.3f}", "degenerate, 100% refusal"])
        except pd.errors.EmptyDataError:
            pass

    for p in PROVIDERS:
        r = e2b_pairwise_jsd(p)
        if r is None:
            rows.append([f"E2b paraphrase floor, {MODEL_LABEL[p]}", "", "", "missing"])
            continue
        rows.append([f"E2b paraphrase floor, {MODEL_LABEL[p]} ({r['domain']}_{r['cell']})",
                     f"{r['median_jsd']:.3f}", f"{r['max_jsd']:.3f}",
                     "non-degenerate, top-fab cell"])

    e3_csv = DATA_AGG / "seed_jsd.csv"
    if e3_csv.exists() and e3_csv.stat().st_size > 0:
        try:
            e3 = pd.read_csv(e3_csv)
            e3_max = e3.jsd_point.max() if not e3.empty else 0
            rows.append(["E3 seed floor (Claude xray D0 three-way split)",
                         "", f"{e3_max:.3f}", ""])
        except pd.errors.EmptyDataError:
            pass

    out = out_dir / "table2_ablations.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "jsd_median", "jsd_max", "notes"])
        w.writerows(rows)
    print(f"wrote {out}")

    ratio_rows = []
    for p in PROVIDERS:
        sub = jsd_df[jsd_df.model == p]
        demo_max = sub.jsd_point.max()
        r = e2b_pairwise_jsd(p)
        floor = r["max_jsd"] if r else None
        ratio = f"{demo_max / floor:.1f}" if floor and floor > 0 else ""
        ratio_rows.append([MODEL_LABEL[p], f"{demo_max:.3f}",
                           f"{floor:.3f}" if floor is not None else "", ratio])

    out2 = out_dir / "table2_ratios.csv"
    with out2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["provider", "demographic_jsd_max", "e2b_paraphrase_floor_max", "ratio"])
        w.writerows(ratio_rows)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
