"""Compressed-at-rest model packs with safe extraction and integrity checks."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Mapping

from .deployment_manifest import make_deployment_manifest, sha256_file


def create_model_pack(
    manifest: Mapping[str, object],
    model_directory: str | Path,
    output_zip: str | Path,
) -> Path:
    """Create a deflated zip for storage/transport, retaining model checksums."""
    models_root = Path(model_directory)
    checked = make_deployment_manifest(manifest["models"], manifest.get("source", {}))
    destination = Path(output_zip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("manifest.json", json.dumps(checked, indent=2, sort_keys=True) + "\n")
        for entry in checked["models"]:
            source = models_root / entry["path"]
            if not source.is_file() or sha256_file(source) != entry["sha256"]:
                raise ValueError(f"Model integrity mismatch before packing: {source}")
            archive.write(source, arcname=f"models/{entry['path']}")
    return destination


def install_model_pack(pack_path: str | Path, cache_directory: str | Path) -> Path:
    """Safely decompress a verified pack into the app-style on-demand cache."""
    pack = Path(pack_path)
    cache = Path(cache_directory)
    with tempfile.TemporaryDirectory(dir=cache.parent if cache.parent.exists() else None) as temp_dir:
        staging = Path(temp_dir) / "pack"
        staging.mkdir(parents=True)
        with zipfile.ZipFile(pack) as archive:
            for member in archive.infolist():
                target = (staging / member.filename).resolve()
                if not str(target).startswith(str(staging.resolve())):
                    raise ValueError("Refusing unsafe archive member path.")
            archive.extractall(staging)
        manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        make_deployment_manifest(manifest["models"], manifest.get("source", {}))
        for entry in manifest["models"]:
            model = staging / "models" / entry["path"]
            if not model.is_file() or sha256_file(model) != entry["sha256"]:
                raise ValueError(f"Pack integrity check failed: {entry['path']}")
        if cache.exists():
            shutil.rmtree(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(cache))
    return cache
