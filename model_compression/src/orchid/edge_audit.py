"""Host-only edge-readiness measurements for frozen orchid checkpoints."""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .checkpoints import load_orchid_checkpoint
from .evaluation import load_bundle_model
from .models import OrchidTaxonomyIndex, build_orchid_model


def load_auditable_model(checkpoint: str | Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load either a paper single-model checkpoint or legacy cascade component."""
    bundle = load_orchid_checkpoint(checkpoint, map_location="cpu")
    metadata = bundle["metadata"]
    if metadata.get("task") == "paper_single_model":
        taxonomy = OrchidTaxonomyIndex.from_species_ids(metadata["class_labels"])
        model = build_orchid_model(metadata["method"], taxonomy, use_imagenet_weights=False)
        model.load_state_dict(bundle["model_state_dict"])
        return model.eval(), metadata
    return load_bundle_model(checkpoint, torch.device("cpu"))


def _primary_output(value: Any) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def audit_models(models: Sequence[torch.nn.Module], image_size: int, *, warmup: int = 10, trials: int = 50) -> dict[str, float | int | str]:
    """Measure serialized parameters and CPU forward calls, never a phone proxy."""
    if not models or image_size <= 0 or warmup < 0 or trials <= 0:
        raise ValueError("models, image_size, and positive trials are required.")
    sample = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
    for model in models:
        model.cpu().eval()
        with torch.no_grad():
            for _ in range(warmup):
                _primary_output(model(sample))
    timings = []
    with torch.no_grad():
        for _ in range(trials):
            started = time.perf_counter()
            for model in models:
                _primary_output(model(sample))
            timings.append((time.perf_counter() - started) * 1000.0)
    parameters = sum(parameter.numel() for model in models for parameter in model.parameters())
    return {
        "measurement_scope": "host_cpu_only_not_mobile_device",
        "input_shape": f"1x3x{image_size}x{image_size}",
        "model_files": len(models),
        "neural_inference_calls_per_input": len(models),
        "parameter_count": int(parameters),
        "host_cpu_latency_ms_p50": float(np.percentile(timings, 50)),
        "host_cpu_latency_ms_p95": float(np.percentile(timings, 95)),
        "host_peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "warmup_calls": warmup,
        "timed_calls": trials,
    }


def checkpoint_audit(
    checkpoints: Sequence[str | Path],
    output: str | Path,
    *,
    runtime_checkpoints: Sequence[str | Path] | None = None,
    runtime_selection: str = "all_packaged_models",
    warmup: int = 10,
    trials: int = 50,
) -> Path:
    """Audit package footprint and the models actually invoked per input."""
    packaged_paths = [Path(path) for path in checkpoints]
    runtime_paths = [Path(path) for path in (runtime_checkpoints or checkpoints)]
    models_and_metadata = [load_auditable_model(checkpoint) for checkpoint in runtime_paths]
    image_sizes = {int(metadata["img_size"]) for _, metadata in models_and_metadata}
    if len(image_sizes) != 1:
        raise ValueError("All runtime models in an edge condition must use the same input size.")
    report = audit_models([model for model, _ in models_and_metadata], image_sizes.pop(), warmup=warmup, trials=trials)
    packaged_parameter_count = 0
    for checkpoint in packaged_paths:
        bundle = load_orchid_checkpoint(checkpoint, map_location="cpu")
        packaged_parameter_count += sum(
            tensor.numel()
            for name, tensor in bundle["model_state_dict"].items()
            if not name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        )
    report["runtime_parameter_count"] = report["parameter_count"]
    report["parameter_count"] = int(packaged_parameter_count)
    report["model_files"] = len(packaged_paths)
    report["neural_inference_calls_per_input"] = len(runtime_paths)
    report["runtime_selection"] = runtime_selection
    report["checkpoint_bytes"] = int(sum(os.path.getsize(path) for path in packaged_paths))
    report["checkpoints"] = [str(path) for path in packaged_paths]
    report["runtime_checkpoints"] = [str(path) for path in runtime_paths]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def litert_parity(torch_logits: np.ndarray, litert_logits: np.ndarray) -> dict[str, float | int]:
    """Compare exported logits; callers must supply identically preprocessed rows."""
    left, right = np.asarray(torch_logits), np.asarray(litert_logits)
    if left.shape != right.shape or left.ndim != 2 or not left.size:
        raise ValueError("Parity requires equally shaped non-empty NxC logits.")
    return {
        "n_examples": int(left.shape[0]),
        "top1_agreement": float(np.mean(left.argmax(axis=1) == right.argmax(axis=1))),
        "max_absolute_logit_delta": float(np.max(np.abs(left - right))),
        "mean_absolute_logit_delta": float(np.mean(np.abs(left - right))),
    }
