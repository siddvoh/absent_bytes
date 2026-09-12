"""Primary analysis (v2): JSD matrix, bootstrap CIs, refusal rates, noise floors.

Inputs: data/categorized/ — v2 records with `response_json` and `is_mirage`
(Asadi judge cross-check). Diagnosis categorization is deterministic via
src.normalize.extract_diagnosis (no LLM judge).

Outputs (all in data/aggregated/):
    jsd_matrix.{csv,parquet}      demographic JSDs vs D0 per (model, domain)
    refusal_rates.{csv,parquet}   per-cell refusal rate + self-reported
                                  image_present rate + Asadi-judge mirage rate
    distributions.{csv,parquet}   per-cell diagnosis distributions
    top5_shifts.json              top-5 category shifts per cell comparison
    paraphrase_jsd.csv            E2 noise floor
    seed_jsd.csv                  E3 noise floor
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metrics import (
    bootstrap_jsd,
    fabrication_rate,
    image_present_rate,
    load_records,
    refusal_rate,
    samples_to_distribution,
    schema_compliance_rate,
    top_k_shifts,
)
from normalize import extract_diagnosis
from utils import DATA_AGG, DATA_CAT, load_taxonomy

MODELS = ["claude-opus-4-7", "gpt-5.4", "gemini-3.1-pro-preview"]
DOMAINS = ["xray", "mri", "derm"]
FACTORIAL = [
    "32wm", "32wf", "32bm", "32bf", "32rm", "32rf",
    "65wm", "65wf", "65bm", "65bf", "65rm", "65rf",
]
DEMOS = ["D0"] + FACTORIAL


def cell_path(model: str, domain: str, demographic: str, experiment: str = "E1") -> Path:
    return DATA_CAT / f"{experiment}_{model}_{domain}_{demographic}.jsonl"


def run_primary_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jsd_rows: list[dict] = []
    refusal_rows: list[dict] = []
    dist_rows: list[dict] = []
    top5_tables: dict[tuple[str, str, str], dict] = {}

    for model, domain in product(MODELS, DOMAINS):
        taxonomy = load_taxonomy(domain)["categories"]
        cats_with_nd = list(dict.fromkeys(list(taxonomy) + ["Other", "NO_DIAGNOSIS"]))
        demog_samples: dict[str, list[str]] = {}

        for demog in DEMOS:
            p = cell_path(model, domain, demog)
            if not p.exists():
                print(f"  MISSING: {p}")
                continue
            records = load_records(p)
            samples = [extract_diagnosis(r, domain) for r in records]
            demog_samples[demog] = samples

            refusal_rows.append({
                "model": model,
                "domain": domain,
                "demographic": demog,
                "n_total": len(records),
                "fabrication_rate": fabrication_rate(records),
                "refusal_rate": refusal_rate(records),
                "image_present_rate": image_present_rate(records),
                "asadi_judge_mirage_rate": (
                    sum(1 for r in records if r.get("is_mirage") is True) / max(len(records), 1)
                ),
                "schema_compliance": schema_compliance_rate(records),
            })

            dist = samples_to_distribution(samples or ["NO_DIAGNOSIS"], cats_with_nd)
            for cat, p_ in zip(cats_with_nd, dist):
                dist_rows.append({
                    "model": model,
                    "domain": domain,
                    "demographic": demog,
                    "category": cat,
                    "proportion": float(p_),
                    "n_samples": len(samples),
                })

        a = demog_samples.get("D0")
        if not a:
            continue
        for d in FACTORIAL:
            b = demog_samples.get(d)
            if not b:
                continue
            bs = bootstrap_jsd(a, b, cats_with_nd, n_bootstrap=1000)
            jsd_rows.append({
                "model": model,
                "domain": domain,
                "comparison": f"{d}_vs_D0",
                "jsd_point": bs["point"],
                "jsd_mean": bs["mean"],
                "ci_lo": bs["ci_lo"],
                "ci_hi": bs["ci_hi"],
                "n_a": len(a),
                "n_b": len(b),
            })
            top5_tables[(model, domain, f"{d}_vs_D0")] = {
                "shifts": top_k_shifts(a, b, cats_with_nd, k=5),
                "n_a": len(a),
                "n_b": len(b),
            }

    jsd_df = pd.DataFrame(jsd_rows)
    ref_df = pd.DataFrame(refusal_rows)
    dist_df = pd.DataFrame(dist_rows)
    DATA_AGG.mkdir(parents=True, exist_ok=True)
    jsd_df.to_parquet(DATA_AGG / "jsd_matrix.parquet")
    jsd_df.to_csv(DATA_AGG / "jsd_matrix.csv", index=False)
    ref_df.to_parquet(DATA_AGG / "refusal_rates.parquet")
    ref_df.to_csv(DATA_AGG / "refusal_rates.csv", index=False)
    dist_df.to_parquet(DATA_AGG / "distributions.parquet")
    dist_df.to_csv(DATA_AGG / "distributions.csv", index=False)

    with open(DATA_AGG / "top5_shifts.json", "w") as f:
        json.dump(
            {f"{m}__{d}__{c}": v for (m, d, c), v in top5_tables.items()},
            f,
            indent=2,
        )

    print("\nJSD matrix:")
    print(jsd_df.to_string(index=False) if not jsd_df.empty else "  (empty)")
    print("\nRefusal / mirage rates:")
    print(ref_df.to_string(index=False) if not ref_df.empty else "  (empty)")
    return jsd_df, ref_df, dist_df


def run_ablations() -> None:
    """E2 paraphrase noise floor (Claude xray D0 vs 3 paraphrases) and
    E3 seed noise floor (three-way split of Claude xray D0)."""
    taxonomy = load_taxonomy("xray")["categories"] + ["Other", "NO_DIAGNOSIS"]
    p0 = cell_path("claude-opus-4-7", "xray", "D0")
    if not p0.exists():
        print("E1 D0 not available; skipping ablations")
        return
    d0 = [extract_diagnosis(r, "xray") for r in load_records(p0)]

    paraphrase_rows: list[dict] = []
    for p_key in ("D0_P1", "D0_P2", "D0_P3"):
        p = DATA_CAT / f"E2_claude-opus-4-7_xray_{p_key}.jsonl"
        if not p.exists():
            continue
        samples = [extract_diagnosis(r, "xray") for r in load_records(p)]
        bs = bootstrap_jsd(d0, samples, taxonomy, n_bootstrap=1000)
        paraphrase_rows.append({
            "comparison": f"D0_vs_{p_key}",
            "jsd_point": bs["point"],
            "jsd_mean": bs["mean"],
            "ci_lo": bs["ci_lo"],
            "ci_hi": bs["ci_hi"],
        })

    rng = np.random.default_rng(20260418)
    d0_arr = np.array(d0, dtype=object)
    rng.shuffle(d0_arr)
    thirds = np.array_split(d0_arr, 3)
    seed_rows: list[dict] = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        bs = bootstrap_jsd(list(thirds[i]), list(thirds[j]), taxonomy, n_bootstrap=1000)
        seed_rows.append({
            "comparison": f"seed_split_{i}_vs_{j}",
            "jsd_point": bs["point"],
            "jsd_mean": bs["mean"],
            "ci_lo": bs["ci_lo"],
            "ci_hi": bs["ci_hi"],
        })

    pdf = pd.DataFrame(paraphrase_rows)
    sdf = pd.DataFrame(seed_rows)
    pdf.to_csv(DATA_AGG / "paraphrase_jsd.csv", index=False)
    sdf.to_csv(DATA_AGG / "seed_jsd.csv", index=False)
    print("\nParaphrase noise floor:")
    print(pdf.to_string(index=False) if not pdf.empty else "  (empty)")
    print("\nSeed noise floor:")
    print(sdf.to_string(index=False) if not sdf.empty else "  (empty)")


def main() -> None:
    run_primary_analysis()
    run_ablations()


if __name__ == "__main__":
    main()
