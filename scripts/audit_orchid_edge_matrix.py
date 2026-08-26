#!/usr/bin/env python3
"""Generate host edge audits for every Orchid paper method and seed."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "flat_ce",
    "flat_balanced_softmax",
    "flat_hsc",
    "dual_head",
    "dual_head_taxonomy_hsc",
    "cascade_top1",
    "cascade_top2",
)


def class_count(checkpoint: Path) -> int:
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return len(bundle["metadata"]["class_labels"])


def condition_checkpoints(experiment: Path, method: str, seed: int) -> tuple[list[Path], list[Path], str]:
    if method == "flat_hsc":
        checkpoint = experiment / "flat_balanced_softmax" / f"seed-{seed}" / "checkpoints" / "best_orchid_model.pt"
        return [checkpoint], [checkpoint], "shared_B1_checkpoint_plus_posthoc_HSC"
    if not method.startswith("cascade_"):
        checkpoint = experiment / method / f"seed-{seed}" / "checkpoints" / "best_orchid_model.pt"
        return [checkpoint], [checkpoint], "single_model_exact"

    cascade = experiment / "cascade_models" / f"seed-{seed}"
    router = cascade / "router" / "checkpoints" / "best_orchid_model.pt"
    experts = sorted((cascade / "experts").glob("*/checkpoints/best_orchid_model.pt"))
    if not router.is_file() or not experts:
        raise FileNotFoundError(f"Incomplete cascade checkpoints for seed {seed}")
    ranked_experts = sorted(experts, key=lambda path: (-class_count(path), str(path)))
    expert_calls = 1 if method == "cascade_top1" else 2
    runtime = [router, *ranked_experts[:expert_calls]]
    selection = f"router_plus_{expert_calls}_largest_expert_heads_conservative_proxy"
    return [router, *experts], runtime, selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default="artifacts/orchid/public-50k/orchid-hsc-paper")
    parser.add_argument("--output-dir", default="artifacts/orchid/edge_audits/all_seeds")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 123])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    experiment = Path(args.experiment_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tasks = [(method, seed) for method in METHODS for seed in args.seeds]
    rows = []
    for method, seed in tqdm(tasks, desc="Edge audits", unit=" condition"):
        packaged, runtime, selection = condition_checkpoints(experiment, method, seed)
        missing = [str(path) for path in packaged if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
        destination = output / f"{method}_seed-{seed}.json"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "audit_orchid_edge.py"),
            "--output", str(destination),
            "--warmup", str(args.warmup),
            "--trials", str(args.trials),
            "--runtime-selection", selection,
        ]
        for checkpoint in packaged:
            command.extend(("--checkpoint", str(checkpoint)))
        for checkpoint in runtime:
            command.extend(("--runtime-checkpoint", str(checkpoint)))
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        report = json.loads(destination.read_text(encoding="utf-8"))
        rows.append({
            "method": method,
            "seed": seed,
            "model_files": report["model_files"],
            "neural_inference_calls_per_input": report["neural_inference_calls_per_input"],
            "parameter_count": report["parameter_count"],
            "checkpoint_bytes": report["checkpoint_bytes"],
            "host_cpu_latency_ms_p50": report["host_cpu_latency_ms_p50"],
            "host_cpu_latency_ms_p95": report["host_cpu_latency_ms_p95"],
            "host_peak_rss_mb": report["host_peak_rss_mb"],
            "runtime_selection": report["runtime_selection"],
        })

    csv_path = output / "edge_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = [
        "# Orchid host edge audits",
        "",
        "These are host CPU measurements, not Android-device measurements. Cascade latency uses the declared conservative largest-head runtime proxy; package footprint includes the router and all 68 experts.",
        "",
        "| Method | Seed | Files | Calls/input | Parameters | Checkpoint MiB | CPU p50 ms | CPU p95 ms | Peak RSS MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            f"| {row['method']} | {row['seed']} | {row['model_files']} | {row['neural_inference_calls_per_input']} | "
            f"{row['parameter_count']} | {row['checkpoint_bytes'] / 2**20:.2f} | "
            f"{row['host_cpu_latency_ms_p50']:.2f} | {row['host_cpu_latency_ms_p95']:.2f} | {row['host_peak_rss_mb']:.2f} |"
        )
    (output / "README.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
