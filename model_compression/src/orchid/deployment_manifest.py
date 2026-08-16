"""Versioned LiteRT deployment metadata with integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEPLOYMENT_MANIFEST_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployment_entry(tflite_path: str | Path, metadata: Mapping[str, Any], role: str, genus_id: str | None = None) -> dict[str, Any]:
    path = Path(tflite_path)
    if role not in {"router", "expert"}:
        raise ValueError("role must be router or expert.")
    if role == "router" and metadata.get("task") != "genus":
        raise ValueError("Router export requires a genus checkpoint.")
    if role == "expert" and (metadata.get("target_genus") != genus_id or not genus_id):
        raise ValueError("Expert export requires its matching genus_id.")
    return {
        "role": role,
        "genus_id": genus_id,
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "task": metadata["task"],
        "class_labels": metadata["class_labels"],
        "model_name": metadata["model_name"],
        "img_size": metadata["img_size"],
        "normalization": metadata["normalization"],
    }


def make_deployment_manifest(entries: Iterable[Mapping[str, Any]], source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    models = [dict(entry) for entry in entries]
    routers = [entry for entry in models if entry["role"] == "router"]
    if len(routers) != 1:
        raise ValueError("A deployable cascade must include exactly one router.")
    experts = [entry for entry in models if entry["role"] == "expert"]
    router_labels = routers[0]["class_labels"]
    for expert in experts:
        if expert["genus_id"] not in router_labels:
            raise ValueError(f"Expert {expert['genus_id']} is absent from router labels.")
        if expert["normalization"] != routers[0]["normalization"] or expert["img_size"] != routers[0]["img_size"]:
            raise ValueError("All deployed models must use identical preprocessing.")
    return {"schema_version": DEPLOYMENT_MANIFEST_VERSION, "source": dict(source or {}), "models": models}


def write_deployment_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
