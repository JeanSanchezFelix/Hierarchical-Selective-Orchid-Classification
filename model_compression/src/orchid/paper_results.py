"""Paired paper statistics from held-out orchid prediction files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def _read_predictions(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image_file", "true_species_id", "true_genus_id", "predicted_species_id", "predicted_genus_id"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"{path} is not a paper prediction file; missing {sorted(required - set(rows[0]) if rows else required)}")
    return rows


def _hierarchical_correct(row: Mapping[str, str]) -> bool:
    level = row.get("decision_level", "species")
    return (level == "species" and row["predicted_species_id"] == row["true_species_id"]) or (
        level == "genus" and row["predicted_genus_id"] == row["true_genus_id"]
    )


def hierarchical_aurc(rows: Iterable[Mapping[str, str]]) -> float:
    """Area under hierarchical risk-coverage; lower is safer."""
    ordered = sorted(rows, key=lambda row: (-float(row.get("confidence") or 0.0), row["image_file"]))
    if not ordered:
        raise ValueError("Cannot compute hAURC for zero predictions.")
    errors = np.asarray([not _hierarchical_correct(row) for row in ordered], dtype=float)
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1)))


def risk_coverage_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, float]]:
    ordered = sorted(rows, key=lambda row: (-float(row.get("confidence") or 0.0), row["image_file"]))
    errors = np.asarray([not _hierarchical_correct(row) for row in ordered], dtype=float)
    return [{"coverage": (index + 1) / len(ordered), "hierarchical_risk": float(errors[: index + 1].mean())} for index in range(len(ordered))]


def paired_bootstrap_hauc_difference(
    candidate_csv: str | Path, reference_csv: str | Path, *, samples: int = 2000, seed: int = 2026
) -> dict[str, float | int]:
    candidate = {row["image_file"]: row for row in _read_predictions(candidate_csv)}
    reference = {row["image_file"]: row for row in _read_predictions(reference_csv)}
    if candidate.keys() != reference.keys():
        raise ValueError("Paired bootstrap requires identical image_file sets.")
    keys = sorted(candidate)
    if any(candidate[key]["true_species_id"] != reference[key]["true_species_id"] for key in keys):
        raise ValueError("Paired bootstrap requires identical ground-truth labels.")
    observed = hierarchical_aurc(candidate.values()) - hierarchical_aurc(reference.values())
    rng = np.random.default_rng(seed)
    deltas = []
    for indices in rng.integers(0, len(keys), size=(samples, len(keys))):
        left = [candidate[keys[index]] for index in indices]
        right = [reference[keys[index]] for index in indices]
        deltas.append(hierarchical_aurc(left) - hierarchical_aurc(right))
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {"n_images": len(keys), "bootstrap_samples": samples, "hAURC_difference": float(observed), "ci95_low": float(low), "ci95_high": float(high)}


def summarize_matrix(runs: Mapping[str, Mapping[int, str | Path]], reference: str, output_dir: str | Path) -> Path:
    """Create seed-level CSV and paired CIs for every method versus reference."""
    if reference not in runs:
        raise ValueError(f"Reference method {reference!r} is absent from runs.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    paired = []
    curves = []
    reference_seeds = set(runs[reference])
    for method, per_seed in runs.items():
        if set(per_seed) != reference_seeds:
            raise ValueError(f"{method} does not have the same seed set as {reference}.")
        for seed, path in sorted(per_seed.items()):
            predictions = _read_predictions(path)
            rows.append({"method": method, "seed": seed, "n_test_images": len(predictions), "hAURC": hierarchical_aurc(predictions)})
            curves.extend({"method": method, "seed": seed, **point} for point in risk_coverage_rows(predictions))
            if method != reference:
                paired.append({"method": method, "seed": seed, **paired_bootstrap_hauc_difference(path, runs[reference][seed])})
    with (output / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "paired_bootstrap.json").write_text(json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "seed", "coverage", "hierarchical_risk"])
        writer.writeheader(); writer.writerows(curves)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row["hAURC"])
    table = ["| Method | hAURC mean +/- sd | Seeds |", "| --- | ---: | ---: |"]
    for method, values in grouped.items():
        table.append(f"| {method} | {np.mean(values):.6f} +/- {np.std(values, ddof=0):.6f} | {len(values)} |")
    (output / "paper_table_hAURC.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return output
