"""Figure 3 (candidate): demographic JSD vs paraphrase noise floor.

Per provider, plots two bars:
  - Demographic max JSD (top-fabrication cell vs D0)
  - E2b paraphrase floor (max pairwise P-P JSD on same top-fab cell)

The E3 seed floor is reported as a footnote (degenerate at 0.000 because
Claude xray D0 is 100% refusal, so any three-way split of 100 identical
refusals yields JSD = 0).

Dashed line at the 0.10 pre-registered threshold.
"""
from __future__ import annotations

import pathlib
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from metrics import bootstrap_jsd, load_records
from normalize import extract_diagnosis
from utils import DATA_AGG, DATA_CAT, REPO_ROOT, load_taxonomy

PROVIDERS = ["claude-opus-4-7", "gpt-5.4", "gemini-3.1-pro-preview"]
MODEL_SHORT = {
    "claude-opus-4-7": "Claude Opus 4.7",
    "gpt-5.4": "GPT-5.4",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}
TOP_CELL = {
    "claude-opus-4-7": ("derm", "65wm"),
    "gpt-5.4": ("xray", "32bm"),
    "gemini-3.1-pro-preview": ("xray", "32bf"),
}
THRESHOLD = 0.10

COLOR_DEMO = "#d62728"
COLOR_PARA = "#1f77b4"


def paraphrase_max(provider: str) -> float:
    domain, cell = TOP_CELL[provider]
    tax = load_taxonomy(domain)["categories"] + ["Other", "NO_DIAGNOSIS"]
    samples = {}
    for p in ("P1", "P2", "P3"):
        path = DATA_CAT / f"E2b_{provider}_{domain}_{cell}_{p}.jsonl"
        samples[p] = [extract_diagnosis(r, domain) for r in load_records(path)]
    jsds = []
    for a, b in combinations(("P1", "P2", "P3"), 2):
        bs = bootstrap_jsd(samples[a], samples[b], tax, n_bootstrap=500)
        jsds.append(bs["point"])
    return max(jsds)


def main():
    jsd_df = pd.read_parquet(DATA_AGG / "jsd_matrix.parquet")
    seed_df = pd.read_csv(DATA_AGG / "seed_jsd.csv")
    seed_floor = float(seed_df["jsd_point"].max()) if not seed_df.empty else 0.0

    rows = []
    for prov in PROVIDERS:
        demo_max = float(jsd_df[jsd_df.model == prov]["jsd_point"].max())
        para_max = paraphrase_max(prov)
        rows.append({
            "provider": MODEL_SHORT[prov],
            "top_cell": f"{TOP_CELL[prov][0]}_{TOP_CELL[prov][1]}",
            "demo_max": demo_max,
            "para_max": para_max,
            "ratio": demo_max / para_max if para_max > 0 else float("inf"),
        })
    _ = seed_floor

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(rows))
    width = 0.34

    demo_vals = [r["demo_max"] for r in rows]
    para_vals = [r["para_max"] for r in rows]

    b1 = ax.bar(x - width / 2, demo_vals, width,
                label="Demographic JSD (max vs neutral)", color=COLOR_DEMO)
    b2 = ax.bar(x + width / 2, para_vals, width,
                label="Paraphrase JSD (max, same demographic)", color=COLOR_PARA)

    for bars, vals in ((b1, demo_vals), (b2, para_vals)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    for i, r in enumerate(rows):
        if r["para_max"] > 0:
            ax.text(i, max(r["demo_max"], r["para_max"]) + 0.08,
                    f"{r['ratio']:.1f}\u00d7",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color="#333")

    ax.axhline(THRESHOLD, linestyle="--", color="black", linewidth=0.8, alpha=0.6)
    ax.text(len(rows) - 0.5, THRESHOLD + 0.01, "pre-reg threshold 0.10",
            ha="right", va="bottom", fontsize=8, alpha=0.7)

    ax.text(0.01, -0.18,
            "Seed floor (E3): 0.000 (degenerate: Claude xray D0 is 100% refusal).",
            transform=ax.transAxes, fontsize=8, color="gray", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['provider']}\n({r['top_cell']})" for r in rows], fontsize=9)
    ax.set_ylabel("JSD (base 2) vs neutral / vs paraphrase")
    ax.set_ylim(0, max(demo_vals) * 1.25)
    ax.set_title(
        "Demographic signal vs paraphrase and seed noise floors, per provider",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = REPO_ROOT / "data" / "derived" / "figures" / "fig3_noise_floors.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    for r in rows:
        print(f"  {r['provider']:18s} demo={r['demo_max']:.3f} para={r['para_max']:.3f} "
              f"ratio={r['ratio']:.2f}x")


if __name__ == "__main__":
    main()
