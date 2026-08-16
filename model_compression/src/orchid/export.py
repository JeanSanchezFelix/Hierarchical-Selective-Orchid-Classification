"""Export validated orchid checkpoint bundles to LiteRT/TFLite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model_compression.src.converters.to_tflite import convert_pytorch_model_to_tflite

from .checkpoints import load_orchid_checkpoint
from .deployment_manifest import deployment_entry
from .evaluation import load_bundle_model


def export_checkpoint_to_litert(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    role: str,
    genus_id: str | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Export one versioned checkpoint and return its manifest entry.

    The output is intentionally not quantized here; quantization choices must be
    evaluated and recorded as separate experiments rather than silently applied.
    """
    runtime_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_bundle_model(checkpoint_path, runtime_device)
    output = Path(output_path)
    example = torch.zeros((1, 3, int(metadata["img_size"]), int(metadata["img_size"])), device=runtime_device)
    convert_pytorch_model_to_tflite(model, (example,), str(output), runtime_device)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"LiteRT export did not produce a file: {output}")
    return deployment_entry(output, metadata, role=role, genus_id=genus_id)
