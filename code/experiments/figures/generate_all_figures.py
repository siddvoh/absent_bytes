"""Regenerate every figure and table the papers print. Run after run_analysis.py."""
from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]

SCRIPTS = [
    "fig2_jsd_heatmap.py",
    "fig3_noise_floors.py",
    "table_fig1_replacement.py",
    "table2_ablations.py",
    "table3_refusal_rates.py",
]

failed = []
for name in SCRIPTS:
    path = pathlib.Path(__file__).resolve().parent / name
    print(f"running {name}")
    if subprocess.run([sys.executable, str(path)], cwd=REPO).returncode != 0:
        failed.append(name)

if failed:
    print("failed: " + ", ".join(failed), file=sys.stderr)
    sys.exit(1)
