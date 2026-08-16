"""Reproducible, task-scoped paths for private orchid experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OrchidArtifactLayout:
    """Path contract for one named experiment, without storing private images."""

    root: Path
    experiment_id: str

    @classmethod
    def create(cls, root: str | Path, experiment_id: str) -> "OrchidArtifactLayout":
        if not experiment_id or any(part in {"", ".", ".."} for part in Path(experiment_id).parts):
            raise ValueError("experiment_id must be a non-empty relative name.")
        layout = cls(Path(root).expanduser().resolve(), experiment_id)
        for directory in (layout.experiment_dir, layout.checkpoints_dir, layout.exports_dir, layout.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return layout

    @property
    def experiment_dir(self) -> Path:
        return self.root / self.experiment_id

    @property
    def checkpoints_dir(self) -> Path:
        return self.experiment_dir / "checkpoints"

    @property
    def exports_dir(self) -> Path:
        return self.experiment_dir / "exports"

    @property
    def reports_dir(self) -> Path:
        return self.experiment_dir / "reports"

    @property
    def taxonomy_path(self) -> Path:
        return self.experiment_dir / "taxonomy.json"

    @property
    def run_metadata_path(self) -> Path:
        return self.experiment_dir / "run_metadata.json"

    def write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
