#!/usr/bin/env python3
"""Two matched demographic contrasts from the prompt grid."""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COUNTS = ROOT / "data" / "aggregated" / "diagnosis_distribution_counts.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "derived" / "figures"

PANELS = (
    {"model": "gpt-5.4", "domain": "xray", "base": "32wm", "contrast": "32bm",
     "title": "GPT-5.4, chest X-ray"},
    {"model": "claude-opus-4-7", "domain": "derm", "base": "65wm", "contrast": "65wf",
     "title": "Claude Opus 4.7, skin mole"},
)

RACE = {"w": "white", "b": "Black", "r": "brown"}
SEX = {"m": "man", "f": "woman"}
BASE_COLOR = "#3d76af"
CONTRAST_COLOR = "#e8a33d"
NO_DIAGNOSIS = "NO_DIAGNOSIS"

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def describe(code: str) -> str:
    return f"{code[:2]}-year-old {RACE[code[2]]} {SEX[code[3]]}"


def changed_axis(base: str, contrast: str) -> str:
    if base[2] != contrast[2]:
        return f"{RACE[base[2]]} to {RACE[contrast[2]]}"
    if base[3] != contrast[3]:
        return f"{SEX[base[3]]} to {SEX[contrast[3]]}"
    return f"{base[:2]} to {contrast[:2]}"


def load(path: Path) -> dict:
    table: dict = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["domain"], row["demographic"])
            table.setdefault(key, {})[row["category"]] = int(row["count"])
    return table


def categories(base: dict, contrast: dict) -> list[str]:
    ordered = OrderedDict()
    for name, count in sorted(base.items(), key=lambda kv: -kv[1]):
        if name != NO_DIAGNOSIS and count:
            ordered[name] = None
    for name, count in sorted(contrast.items(), key=lambda kv: -kv[1]):
        if name != NO_DIAGNOSIS and count:
            ordered.setdefault(name, None)
    names = list(ordered)
    if NO_DIAGNOSIS in base or NO_DIAGNOSIS in contrast:
        names.append(NO_DIAGNOSIS)
    return names


def draw(ax, panel: dict, table: dict, show_yticklabels: bool) -> None:
    base = table[(panel["model"], panel["domain"], panel["base"])]
    contrast = table[(panel["model"], panel["domain"], panel["contrast"])]
    names = categories(base, contrast)
    positions = range(len(names))
    width = 0.38

    for offset, cells, color, hatch in (
        (-width / 2, base, BASE_COLOR, None),
        (width / 2, contrast, CONTRAST_COLOR, "///"),
    ):
        values = [cells.get(name, 0) for name in names]
        bars = ax.bar([p + offset for p in positions], values, width,
                      color=color, hatch=hatch, edgecolor="#333333", linewidth=0.45)
        for rect, value in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + 2.5, str(value),
                    ha="center", va="bottom", fontsize=7.1)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(["No diagnosis" if n == NO_DIAGNOSIS else n for n in names], fontsize=7.1)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    if show_yticklabels:
        ax.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=7.1)
        ax.set_ylabel("Responses (out of 100)", fontsize=7.1)
    else:
        ax.set_yticklabels([])
    ax.yaxis.grid(True, color="#cccccc", linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(length=0)

    ax.set_title(f"({panel['letter']}) {panel['title']}", fontsize=8.6, fontweight="bold",
                 loc="left", pad=16)
    ax.text(0, 1.015, f"The prompt changes only {changed_axis(panel['base'], panel['contrast'])}",
            transform=ax.transAxes, fontsize=6.9, color="#555555")
    ax.legend(handles=[
        Patch(facecolor=BASE_COLOR, edgecolor="#333333", label=describe(panel["base"])),
        Patch(facecolor=CONTRAST_COLOR, edgecolor="#333333", hatch="///",
              label=describe(panel["contrast"])),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False, fontsize=7.1)


def build(counts: Path, out_dir: Path) -> None:
    table = load(counts)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.13))
    for letter, (ax, panel) in zip("ab", zip(axes, PANELS)):
        draw(ax, dict(panel, letter=letter), table, show_yticklabels=(letter == "a"))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.20, wspace=0.16)
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"absent_byte_demographic_pairs.{suffix}", dpi=300)
    plt.close(fig)
    print(out_dir / "absent_byte_demographic_pairs.pdf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    build(args.counts, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
