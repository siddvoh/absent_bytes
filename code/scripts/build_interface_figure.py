#!/usr/bin/env python3
"""The audited path and the client check, drawn from the end-to-end trace."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE = ROOT / "data" / "evidence" / "end_to_end_trace.json"
DEFAULT_OUT_DIR = ROOT / "data" / "derived" / "figures"

EDGE = "#555555"
OBSERVED_EDGE = "#b5651d"
OBSERVED_FILL = "#fdf1e0"
GUARDED_EDGE = "#2e7d6f"
GUARDED_FILL = "#e6f2ef"

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

COLUMNS = ((0.147, 0.226), (0.408, 0.282), (0.722, 0.258))
ROWS = ((0.613, 0.241), (0.205, 0.241))


def claimed_attachments(trace: dict) -> str:
    match = re.search(r"number of image attachments:\s*(\d+)", trace["rendered_prompt"])
    return match.group(1) if match else "?"


def panels(trace: dict) -> tuple[list[str], list[str]]:
    audit = trace["serialized_payload_audit"]
    guarded = trace["guarded_counterpart"]
    observed = [
        'Prompt text says\n"number of image\nattachments: %s"' % claimed_attachments(trace),
        "Retained client construction state\n%d image parts, %d image bytes"
        % (len(audit["image_parts"]), audit["serialized_image_byte_count"]),
        "Returned diagnosis field\n%s" % trace["parsed_response"]["primary_diagnosis"],
    ]
    blocked = guarded["response"]["primary_diagnosis"] is None
    counterpart = [
        "Same prompt and\nretained client state",
        "Client compares attachment claim\nwith image parts and\nbyte count",
        "Mismatch found\nCall blocked before dispatch\nDiagnosis remains %s"
        % ("null" if blocked else "set"),
    ]
    return observed, counterpart


def draw_row(fig, texts, y, height, label, accent_edge, accent_fill) -> None:
    fig.text(0.020, y + height / 2, label, fontsize=8.8, fontweight="bold",
             va="center", ha="left", color="#222222")
    for index, ((x, width), text) in enumerate(zip(COLUMNS, texts)):
        last = index == len(COLUMNS) - 1
        fig.patches.append(Rectangle(
            (x, y), width, height, transform=fig.transFigure,
            facecolor=accent_fill if last else "white",
            edgecolor=accent_edge if last else EDGE,
            linewidth=1.1 if last else 0.9))
        fig.text(x + width / 2, y + height / 2, text, fontsize=8.4,
                 ha="center", va="center", color="#111111", linespacing=1.45)
        if not last:
            start = x + width + 0.006
            end = COLUMNS[index + 1][0] - 0.004
            fig.add_artist(FancyArrowPatch(
                (start, y + height / 2), (end, y + height / 2),
                transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9,
                linewidth=1.1, color="#222222", shrinkA=0, shrinkB=0))


def build(trace_path: Path, out_dir: Path) -> None:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    observed, counterpart = panels(trace)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.08, 2.21))
    draw_row(fig, observed, ROWS[0][0], ROWS[0][1], "Observed\nrecord",
             OBSERVED_EDGE, OBSERVED_FILL)
    draw_row(fig, counterpart, ROWS[1][0], ROWS[1][1], "Guarded\ncounterpart",
             GUARDED_EDGE, GUARDED_FILL)
    fig.add_artist(plt.Line2D([0.022, 0.980], [0.519, 0.519], transform=fig.transFigure,
                              color="#d8d8d8", linewidth=0.9))
    fig.text(0.147, 0.085,
             "Retained client state is author-side evidence. "
             "It does not attest to provider receipt.",
             fontsize=8.2, style="italic", color="#666666", ha="left")
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"absent_byte_interface_flow.{suffix}", dpi=300)
    plt.close(fig)
    print(out_dir / "absent_byte_interface_flow.pdf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    build(args.trace, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
