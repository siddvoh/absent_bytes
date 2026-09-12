"""Figure 2: JSD heatmap — 9 (model, domain) rows × 12 factorial demographics."""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from utils import DATA_AGG, REPO_ROOT

MODEL_ORDER = ["claude-opus-4-7", "gpt-5.4", "gemini-3.1-pro-preview"]
DOMAIN_ORDER = ["xray", "mri", "derm"]
FACTORIAL = ["32wm", "32wf", "32bm", "32bf", "32rm", "32rf",
             "65wm", "65wf", "65bm", "65bf", "65rm", "65rf"]
MODEL_SHORT = {
    "claude-opus-4-7": "Claude Opus 4.7",
    "gpt-5.4": "GPT-5.4",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}
DOMAIN_SHORT = {"xray": "X-ray", "mri": "MRI", "derm": "Derm"}


def main():
    df = pd.read_parquet(DATA_AGG / "jsd_matrix.parquet")
    df["demog"] = df["comparison"].str.replace("_vs_D0", "", regex=False)
    df["row_label"] = df.apply(lambda r: f"{MODEL_SHORT[r.model]} / {DOMAIN_SHORT[r.domain]}", axis=1)
    row_order = [
        f"{MODEL_SHORT[m]} / {DOMAIN_SHORT[d]}"
        for m in MODEL_ORDER for d in DOMAIN_ORDER
    ]
    pivot = df.pivot_table(index="row_label", columns="demog", values="jsd_point").reindex(row_order)[FACTORIAL]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="Reds",
        vmin=0, vmax=max(0.5, float(pivot.values.max())),
        cbar_kws={"label": "JSD (base 2) vs neutral baseline"},
        linewidths=0.5, linecolor="white", annot_kws={"fontsize": 8},
        ax=ax,
    )
    ax.set_xticklabels(FACTORIAL, rotation=0, fontsize=9)
    ax.set_ylabel("")
    ax.set_xlabel("Demographic condition ({age}{race}{sex}: w=white b=Black r=brown, m=man f=woman)", fontsize=9)
    ax.set_title("Demographic-vs-neutral JSD across 9 (model, domain) cells × 12 factorial demographics", fontsize=11, pad=12)
    plt.tight_layout()
    out = REPO_ROOT / "data" / "derived" / "figures" / "fig2_jsd_heatmap.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
