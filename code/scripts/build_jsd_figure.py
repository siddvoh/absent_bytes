"""Build the JSD heatmap manuscript figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "aggregated" / "jsd_matrix_six_model.parquet"
DEFAULT_OUT_DIR = ROOT / "data" / "derived" / "figures"

MODEL_ORDER = [
    "claude-opus-4-7",
    "gpt-5.4",
    "gemini-3.1-pro-preview",
    "qwen3-vl-32b-instruct",
    "llama-4-maverick",
    "medgemma-4b-it",
]
DOMAIN_ORDER = ["xray", "mri", "derm"]
FACTORIAL = [
    "32wm", "32wf", "32bm", "32bf", "32rm", "32rf",
    "65wm", "65wf", "65bm", "65bf", "65rm", "65rf",
]
MODEL_SHORT = {
    "claude-opus-4-7": "Claude Opus 4.7",
    "gpt-5.4": "GPT-5.4",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "qwen3-vl-32b-instruct": "Qwen3-VL 32B",
    "llama-4-maverick": "Llama 4 Maverick",
    "medgemma-4b-it": "MedGemma 4B-IT",
}
DOMAIN_SHORT = {"xray": "X-ray", "mri": "MRI", "derm": "Derm"}


def build(data_path: Path, out_dir: Path) -> None:
    frame = pd.read_parquet(data_path)
    frame["demog"] = frame["comparison"].str.replace("_vs_D0", "", regex=False)
    frame["row_label"] = frame.apply(
        lambda row: f"{MODEL_SHORT[row.model]} / {DOMAIN_SHORT[row.domain]}", axis=1
    )
    row_order = [
        f"{MODEL_SHORT[model]} / {DOMAIN_SHORT[domain]}"
        for model in MODEL_ORDER
        for domain in DOMAIN_ORDER
    ]
    pivot = frame.pivot_table(
        index="row_label", columns="demog", values="jsd_point"
    ).reindex(row_order)[FACTORIAL]
    values = pivot.to_numpy(dtype=float)
    vmax = max(0.5, float(np.nanmax(values)))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(12, 5.0))
    image = ax.imshow(values, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("JSD (base 2) vs neutral baseline")

    ax.set_yticks(np.arange(len(pivot.index)), pivot.index, fontsize=7)
    ax.set_xticks(np.arange(len(FACTORIAL)), FACTORIAL, fontsize=7)
    ax.set_xticks(np.arange(-0.5, len(FACTORIAL), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = vmax * 0.55
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            label = "" if np.isnan(value) else f"{value:.2f}"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value > threshold else "black",
            )
    ax.set_ylabel("")
    ax.set_xlabel(
        "{age}{race}{sex}: w=white, b=Black, r=brown, m=man, f=woman",
        fontsize=8,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "jsd_heatmap_six_model.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "jsd_heatmap_six_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    build(args.data, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
