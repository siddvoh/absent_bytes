"""Compact table that replaces Figure 1.

For each (model, domain), show the top-1 diagnosis (and its proportion) under
four representative demographic conditions: D0, 32bm, 32wf, 65wm.

Outputs:
  data/derived/tables/table_fig1_replacement.tex  (LaTeX tabular body, no \\begin{table})
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from utils import DATA_AGG, REPO_ROOT

MODELS = ["claude-opus-4-7", "gpt-5.4", "gemini-3.1-pro-preview"]
DOMAINS = ["xray", "mri", "derm"]
DEMOS = ["D0",
         "32wm", "32wf", "32bm", "32bf", "32rm", "32rf",
         "65wm", "65wf", "65bm", "65bf", "65rm", "65rf"]
MODEL_SHORT = {
    "claude-opus-4-7": "Claude",
    "gpt-5.4": "GPT-5.4",
    "gemini-3.1-pro-preview": "Gemini",
}
DOMAIN_SHORT = {"xray": "X-ray", "mri": "MRI", "derm": "Derm"}
DEMO_LABEL = {d: ("D0" if d == "D0" else d) for d in DEMOS}


ABBREV = {
    "Sarcoidosis": "Sarc",
    "Neurosarcoidosis": "Nsarc",
    "Neurocysticercosis": "Ncyst",
    "Melanoma": "Mel",
    "Benign Nevus": "BNev",
    "Seborrheic Keratosis": "SebK",
    "Hiatal Hernia": "HHrn",
    "Pneumothorax": "Pnx",
    "Pulmonary Edema": "PEd",
    "Pneumonia": "Pna",
    "Meningioma": "Men",
    "Glioma": "Gli",
    "Atrophy": "Atr",
    "Normal": "Nor",
    "Other": "Oth",
}


def top_entry(sub: pd.DataFrame) -> str:
    if sub.empty:
        return "--"
    hit = sub.sort_values("proportion", ascending=False).iloc[0]
    prop = float(hit["proportion"])
    if prop == 0:
        return "--"
    name = str(hit["category"]).replace("_", " ")
    if name == "NO DIAGNOSIS":
        return "--"
    return f"{ABBREV.get(name, name)} {prop * 100:.0f}"


def main():
    dist = pd.read_parquet(DATA_AGG / "distributions.parquet")

    tex_lines = [
        "\\toprule",
        "Model & Domain & " + " & ".join(DEMO_LABEL[d].replace(" ", "~") for d in DEMOS) + " \\\\",
        "\\midrule",
    ]
    for m in MODELS:
        for d in DOMAINS:
            cells = [MODEL_SHORT[m], DOMAIN_SHORT[d]]
            for demo in DEMOS:
                sub = dist[(dist.model == m) & (dist.domain == d)
                           & (dist.demographic == demo) & (dist.category != "NO_DIAGNOSIS")]
                if sub.empty or sub["proportion"].max() == 0:
                    all_sub = dist[(dist.model == m) & (dist.domain == d)
                                   & (dist.demographic == demo)]
                    cells.append(top_entry(all_sub))
                else:
                    cells.append(top_entry(sub))
            tex_lines.append(" & ".join(cells) + " \\\\")
    tex_lines.append("\\bottomrule")

    out_tex = REPO_ROOT / "data" / "derived" / "tables" / "table_fig1_replacement.tex"
    out_tex.write_text("\n".join(tex_lines))
    print(f"\nwrote {out_tex}")


if __name__ == "__main__":
    main()
