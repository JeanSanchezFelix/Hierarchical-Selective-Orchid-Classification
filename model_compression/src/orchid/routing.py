"""Deterministic genus-to-species cascade routing and probability fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .constants import TASK_GENUS, TASK_GENUS_SPECIES, TASK_TARGET_GENUS


def softmax(logits: Sequence[float]) -> np.ndarray:
    """Numerically stable softmax for one classifier output vector."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Expected one non-empty finite logit vector.")
    shifted = values - values.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


@dataclass(frozen=True)
class GenusCandidate:
    genus_id: str
    router_probability: float


@dataclass(frozen=True)
class SpeciesCandidate:
    species_id: str
    genus_id: str
    expert_probability: float
    joint_probability: float
    fused_probability: float


@dataclass(frozen=True)
class CascadeResult:
    """All information needed to audit a top-k routing decision."""

    routed_genera: tuple[GenusCandidate, ...]
    species_candidates: tuple[SpeciesCandidate, ...]
    selected_species: SpeciesCandidate | None
    fallback_reason: str | None


def validate_cascade_metadata(
    router_metadata: Mapping[str, object],
    expert_metadata_by_genus: Mapping[str, Mapping[str, object]],
) -> None:
    """Reject incompatible model bundles before they reach a mobile device."""
    if router_metadata.get("task") != TASK_GENUS:
        raise ValueError("The router checkpoint must use task='genus'.")
    router_labels = router_metadata.get("class_labels")
    if not isinstance(router_labels, list) or not router_labels:
        raise ValueError("Router metadata must contain non-empty class_labels.")
    router_normalization = router_metadata.get("normalization")
    for genus_id, metadata in expert_metadata_by_genus.items():
        if genus_id not in router_labels:
            raise ValueError(f"Expert '{genus_id}' is not a router label.")
        if metadata.get("task") not in {TASK_GENUS_SPECIES, TASK_TARGET_GENUS}:
            raise ValueError(f"Expert '{genus_id}' does not declare a genus-species task.")
        if metadata.get("target_genus") != genus_id:
            raise ValueError(f"Expert '{genus_id}' has mismatched target_genus metadata.")
        labels = metadata.get("class_labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Expert '{genus_id}' must contain non-empty class_labels.")
        prefix = f"{genus_id}::"
        if any(not isinstance(label, str) or not label.startswith(prefix) for label in labels):
            raise ValueError(f"Expert '{genus_id}' includes a label outside its genus.")
        if metadata.get("normalization") != router_normalization:
            raise ValueError(f"Expert '{genus_id}' uses incompatible normalization.")


class HierarchicalCascadeRouter:
    """Fuse genus and conditional-species probabilities for a fixed model pack.

    The caller supplies logits already produced by the router and any selected
    experts. Keeping neural inference outside this class makes the exact same
    decision rule testable in PyTorch, LiteRT, and Android code.
    """

    def __init__(
        self,
        router_labels: Sequence[str],
        expert_labels_by_genus: Mapping[str, Sequence[str]],
    ) -> None:
        if not router_labels or len(set(router_labels)) != len(router_labels):
            raise ValueError("Router labels must be non-empty and unique.")
        self.router_labels = tuple(router_labels)
        self.expert_labels_by_genus = {genus: tuple(labels) for genus, labels in expert_labels_by_genus.items()}
        for genus, labels in self.expert_labels_by_genus.items():
            if genus not in self.router_labels:
                raise ValueError(f"Expert '{genus}' is not a router class.")
            if not labels or any(not label.startswith(f"{genus}::") for label in labels):
                raise ValueError(f"Expert labels for '{genus}' must be its qualified species IDs.")

    @classmethod
    def from_metadata(
        cls,
        router_metadata: Mapping[str, object],
        expert_metadata_by_genus: Mapping[str, Mapping[str, object]],
    ) -> "HierarchicalCascadeRouter":
        validate_cascade_metadata(router_metadata, expert_metadata_by_genus)
        return cls(
            router_metadata["class_labels"],  # validated above
            {genus: metadata["class_labels"] for genus, metadata in expert_metadata_by_genus.items()},
        )

    def select_genera(self, router_logits: Sequence[float], top_k: int = 2) -> tuple[GenusCandidate, ...]:
        if not 1 <= top_k <= len(self.router_labels):
            raise ValueError(f"top_k must be between 1 and {len(self.router_labels)}.")
        probabilities = softmax(router_logits)
        if probabilities.size != len(self.router_labels):
            raise ValueError("Router logit count does not match router labels.")
        # mergesort makes equal-score ordering deterministic with model label order.
        selected_indices = np.argsort(-probabilities, kind="mergesort")[:top_k]
        return tuple(GenusCandidate(self.router_labels[index], float(probabilities[index])) for index in selected_indices)

    def route(
        self,
        router_logits: Sequence[float],
        expert_logits_by_genus: Mapping[str, Sequence[float]],
        top_k: int = 2,
    ) -> CascadeResult:
        """Route through available top-k experts and return normalized fused candidates.

        For species ``s`` in selected genus ``g``, the unnormalized score is
        ``P(g | image) * P(s | g, image)``. Scores are normalized only over the
        actually evaluated experts, so a missing specialist cannot masquerade as
        evidence against the remaining one.
        """
        routed = self.select_genera(router_logits, top_k=top_k)
        raw_candidates: list[SpeciesCandidate] = []
        missing_experts: list[str] = []
        for genus in routed:
            logits = expert_logits_by_genus.get(genus.genus_id)
            labels = self.expert_labels_by_genus.get(genus.genus_id)
            if logits is None or labels is None:
                missing_experts.append(genus.genus_id)
                continue
            probabilities = softmax(logits)
            if probabilities.size != len(labels):
                raise ValueError(f"Expert '{genus.genus_id}' logit count does not match its labels.")
            raw_candidates.extend(
                SpeciesCandidate(
                    species_id=label,
                    genus_id=genus.genus_id,
                    expert_probability=float(probability),
                    joint_probability=float(genus.router_probability * probability),
                    fused_probability=float(genus.router_probability * probability),
                )
                for label, probability in zip(labels, probabilities)
            )
        if not raw_candidates:
            reason = "no_expert_for_selected_genus" if missing_experts else "no_species_candidates"
            return CascadeResult(routed, (), None, reason)

        total = sum(candidate.joint_probability for candidate in raw_candidates)
        candidates = tuple(
            SpeciesCandidate(
                candidate.species_id,
                candidate.genus_id,
                candidate.expert_probability,
                candidate.joint_probability,
                candidate.joint_probability / total,
            )
            for candidate in raw_candidates
        )
        ordered = tuple(sorted(candidates, key=lambda item: (-item.fused_probability, item.species_id)))
        fallback_reason = None
        if missing_experts:
            fallback_reason = f"missing_expert:{','.join(missing_experts)}"
        return CascadeResult(routed, ordered, ordered[0], fallback_reason)
