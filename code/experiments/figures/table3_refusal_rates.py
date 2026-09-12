"""Table 3: per-cell fabrication/refusal/image_present rates + chi-squared.

Refusal rates and chi-squared over the demographic factorial.
(fabrication_rate, refusal_rate, image_present_rate, asadi_judge_mirage_rate).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from utils import DATA_AGG, REPO_ROOT


def main():
    df = pd.read_csv(DATA_AGG / "refusal_rates.csv")
    out_dir = REPO_ROOT / "data" / "derived" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    chi_rows = []
    for (model, domain), grp in df.groupby(["model", "domain"]):
        grp = grp.copy()
        grp["n_refuse"]    = (grp["n_total"] * grp["refusal_rate"]).round().astype(int)
        grp["n_fabricate"] = grp["n_total"] - grp["n_refuse"]
        mat = grp[["n_refuse", "n_fabricate"]].values
        try:
            chi2, p, _, _ = chi2_contingency(mat)
        except ValueError:
            chi2, p = np.nan, np.nan
        chi_rows.append({
            "model": model, "domain": domain,
            "chi2": chi2, "p_value": p, "n_cells": len(grp),
        })

    chi_df = pd.DataFrame(chi_rows)
    n_tests = len(chi_df)
    chi_df["p_bonf"] = (chi_df["p_value"] * n_tests).clip(0, 1)
    chi_df.to_csv(out_dir / "table3_refusal_chi2.csv", index=False)

    pivot = df.pivot_table(index=["model", "domain"], columns="demographic",
                           values="fabrication_rate").round(3)
    pivot.to_csv(out_dir / "table3_fabrication_pivot.csv")

    md = ["# Table 3: Fabrication rates + refusal-independence chi-squared", ""]
    md.append("## Fabrication rate per cell (primary_diagnosis not null)")
    md.append("")
    md.append(pivot.to_markdown())
    md.append("")
    md.append("## Chi-squared: refusal ~ demographic, per (model, domain)")
    md.append("")
    md.append("Contingency table: rows = (refuse, fabricate), columns = demographics.")
    md.append("")
    md.append(chi_df.to_markdown(index=False, floatfmt=".2e"))
    print("\n".join(md))


if __name__ == "__main__":
    main()
