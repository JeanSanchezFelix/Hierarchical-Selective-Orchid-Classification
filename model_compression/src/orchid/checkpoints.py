"""Self-describing checkpoint bundles for router and genus-expert models."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

from model_compression.src.utils.callbacks.callbacks import Callback


CHECKPOINT_SCHEMA_VERSION = 1


def save_orchid_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    metadata: Mapping[str, Any],
    *,
    epoch: int,
    monitored_metric: float | None,
    optimizer: torch.optim.Optimizer | None = None,
    history: Mapping[str, list[float]] | None = None,
) -> Path:
    """Save weights and deployment-critical provenance in one checkpoint bundle."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": deepcopy(model.state_dict()),
        "optimizer_state_dict": deepcopy(optimizer.state_dict()) if optimizer is not None else None,
        "metadata": dict(metadata),
        "epoch": int(epoch),
        "monitored_metric": float(monitored_metric) if monitored_metric is not None else None,
        "history": dict(history or {}),
    }
    torch.save(payload, destination)
    return destination


def load_orchid_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and minimally validate a versioned orchid checkpoint bundle."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {"schema_version", "model_state_dict", "metadata", "epoch", "monitored_metric"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} is not an orchid checkpoint bundle; missing {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported orchid checkpoint schema: {payload['schema_version']}")
    metadata = payload["metadata"]
    for key in ("task", "class_labels", "model_name", "img_size", "normalization"):
        if key not in metadata:
            raise ValueError(f"Checkpoint metadata is missing '{key}'.")
    return payload


class OrchidModelCheckpoint(Callback):
    """Save the lowest-validation-loss model in the portable orchid bundle format."""

    def __init__(self, save_path: str | Path, metadata: Mapping[str, Any], monitor: str = "val_loss", mode: str = "min"):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        self.save_path = Path(save_path)
        self.metadata = dict(metadata)
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = current < self.best_score if self.mode == "min" else current > self.best_score
        if not improved:
            return
        self.best_score = float(current)
        save_orchid_checkpoint(
            self.save_path,
            logs["model"],
            self.metadata,
            epoch=epoch,
            monitored_metric=self.best_score,
            optimizer=logs.get("optimizer"),
            history=logs.get("history"),
        )
