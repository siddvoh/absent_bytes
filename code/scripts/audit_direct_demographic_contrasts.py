#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MASTER_SEED = 20260418
BOOTSTRAP_DRAWS = 1000


def fail(message: str) -> None:
    raise SystemExit(message)


def read_cells(path: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    cells: dict[tuple[str, str, str], dict[str, object]] = defaultdict(
        lambda: {"counts": {}, "source_sha256": None, "n": None}
    )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        key = (row["model"], row["domain"], row["demographic"])
        cell = cells[key]
        category = row["category"]
        if category in cell["counts"]:
            fail(f"duplicate category in {key}: {category}")
        cell["counts"][category] = int(row["count"])
        n = int(row["n_samples"])
        if cell["n"] not in (None, n):
            fail(f"inconsistent denominator in {key}")
        cell["n"] = n
        digest = row["source_sha256"]
        if cell["source_sha256"] not in (None, digest):
            fail(f"inconsistent source commitment in {key}")
        cell["source_sha256"] = digest
    if len(rows) != 412 or len(cells) != 234:
        fail("diagnosis-count inventory mismatch")
    for key, cell in cells.items():
        if cell["n"] != 100 or sum(cell["counts"].values()) != 100:
            fail(f"cell denominator mismatch: {key}")
    return dict(cells)


def js_divergence(left: dict[str, int], right: dict[str, int]) -> float:
    left_n = sum(left.values())
    right_n = sum(right.values())
    total = 0.0
    for category in set(left) | set(right):
        p = left.get(category, 0) / left_n
        q = right.get(category, 0) / right_n
        midpoint = (p + q) / 2
        if p:
            total += 0.5 * p * math.log2(p / midpoint)
        if q:
            total += 0.5 * q * math.log2(q / midpoint)
    return total


def expand(counts: dict[str, int]) -> np.ndarray:
    return np.asarray(
        [category for category, count in sorted(counts.items()) for _ in range(count)],
        dtype=object,
    )


def resampled_counts(values: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    sampled = values[indices]
    categories, counts = np.unique(sampled, return_counts=True)
    return {str(category): int(count) for category, count in zip(categories, counts)}


def contrast_specs(
    models: list[str], domains: list[str]
) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for model in models:
        for domain in domains:
            for age in ("32", "65"):
                for sex in ("m", "f"):
                    specs.append(
                        {
                            "model": model,
                            "domain": domain,
                            "contrast_type": "white_vs_Black",
                            "fixed_age": age,
                            "fixed_race": "",
                            "fixed_sex": sex,
                            "profile_a": f"{age}w{sex}",
                            "profile_b": f"{age}b{sex}",
                        }
                    )
            for age in ("32", "65"):
                for race in ("w", "b", "r"):
                    specs.append(
                        {
                            "model": model,
                            "domain": domain,
                            "contrast_type": "man_vs_woman",
                            "fixed_age": age,
                            "fixed_race": race,
                            "fixed_sex": "",
                            "profile_a": f"{age}{race}m",
                            "profile_b": f"{age}{race}f",
                        }
                    )
            for race in ("w", "b", "r"):
                for sex in ("m", "f"):
                    specs.append(
                        {
                            "model": model,
                            "domain": domain,
                            "contrast_type": "age_32_vs_65",
                            "fixed_age": "",
                            "fixed_race": race,
                            "fixed_sex": sex,
                            "profile_a": f"32{race}{sex}",
                            "profile_b": f"65{race}{sex}",
                        }
                    )
    return specs


def compute(
    counts_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    cells = read_cells(counts_path)
    models = sorted({key[0] for key in cells})
    domains = sorted({key[1] for key in cells})
    specs = contrast_specs(models, domains)
    if len(specs) != 288:
        fail("direct contrast inventory mismatch")

    details: list[dict[str, str]] = []
    grouped_bootstrap: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    grouped_points: dict[tuple[str, str], list[float]] = defaultdict(list)

    for index, spec in enumerate(specs):
        key_a = (spec["model"], spec["domain"], spec["profile_a"])
        key_b = (spec["model"], spec["domain"], spec["profile_b"])
        if key_a not in cells or key_b not in cells:
            fail(f"missing direct contrast cell: {key_a} or {key_b}")
        left = cells[key_a]
        right = cells[key_b]
        point = js_divergence(left["counts"], right["counts"])
        rng = np.random.default_rng(MASTER_SEED + index)
        left_values = expand(left["counts"])
        right_values = expand(right["counts"])
        bootstrap = np.empty(BOOTSTRAP_DRAWS)
        for draw in range(BOOTSTRAP_DRAWS):
            left_indices = rng.integers(0, len(left_values), size=len(left_values))
            right_indices = rng.integers(0, len(right_values), size=len(right_values))
            bootstrap[draw] = js_divergence(
                resampled_counts(left_values, left_indices),
                resampled_counts(right_values, right_indices),
            )
        group = (spec["model"], spec["contrast_type"])
        grouped_points[group].append(point)
        grouped_bootstrap[group].append(bootstrap)
        details.append(
            {
                **spec,
                "jsd_point": f"{point:.12f}",
                "jsd_bootstrap_mean": f"{float(bootstrap.mean()):.12f}",
                "ci_lo": f"{float(np.percentile(bootstrap, 2.5)):.12f}",
                "ci_hi": f"{float(np.percentile(bootstrap, 97.5)):.12f}",
                "n_a": str(left["n"]),
                "n_b": str(right["n"]),
                "source_sha256_a": str(left["source_sha256"]),
                "source_sha256_b": str(right["source_sha256"]),
            }
        )

    summary: list[dict[str, str]] = []
    for group in sorted(grouped_points):
        point_values = np.asarray(grouped_points[group], dtype=float)
        bootstrap_matrix = np.vstack(grouped_bootstrap[group])
        mean_bootstrap = bootstrap_matrix.mean(axis=0)
        summary.append(
            {
                "model": group[0],
                "contrast_type": group[1],
                "n_comparisons": str(len(point_values)),
                "jsd_mean_point": f"{float(point_values.mean()):.12f}",
                "jsd_bootstrap_mean": f"{float(mean_bootstrap.mean()):.12f}",
                "ci_lo": f"{float(np.percentile(mean_bootstrap, 2.5)):.12f}",
                "ci_hi": f"{float(np.percentile(mean_bootstrap, 97.5)):.12f}",
                "nonzero_comparisons": str(int((point_values > 0).sum())),
            }
        )
    if len(summary) != 18:
        fail("direct contrast summary inventory mismatch")
    return details, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify(
    counts_path: Path,
    details_path: Path,
    summary_path: Path,
) -> None:
    details, summary = compute(counts_path)
    if read_csv(details_path) != details:
        fail("direct demographic contrast detail differs from deterministic rebuild")
    if read_csv(summary_path) != summary:
        fail("direct demographic contrast summary differs from deterministic rebuild")
    print(
        "direct demographic contrasts: 72 white-vs-Black, "
        "108 man-vs-woman, and 108 age-32-vs-65 comparisons verified"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts",
        type=Path,
        default=DATA / "aggregated" / "diagnosis_distribution_counts.csv",
    )
    parser.add_argument(
        "--details",
        type=Path,
        default=DATA / "aggregated" / "direct_demographic_contrasts.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DATA / "aggregated" / "direct_demographic_contrast_summary.csv",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        details, summary = compute(args.counts)
        write_csv(args.details, details)
        write_csv(args.summary, summary)
    verify(args.counts, args.details, args.summary)


if __name__ == "__main__":
    main()
