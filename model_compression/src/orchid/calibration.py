"""Validation-set calibration and abstention rules for the orchid cascade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .routing import CascadeResult, softmax
from .models import OrchidTaxonomyIndex, aggregate_species_probabilities


@dataclass(frozen=True)
class TemperatureScaler:
    temperature: float = 1.0

    def transform(self, logits: Sequence[float]) -> np.ndarray:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        return softmax(np.asarray(logits, dtype=np.float64) / self.temperature)

    @classmethod
    def fit(cls, logits: np.ndarray, targets: Sequence[int], candidates: int = 200) -> "TemperatureScaler":
        """Fit a scalar temperature by validation negative log likelihood."""
        values = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.int64)
        if values.ndim != 2 or values.shape[0] != labels.size or values.shape[0] == 0:
            raise ValueError("logits must be N×C with one target per row.")
        if np.any(labels < 0) or np.any(labels >= values.shape[1]):
            raise ValueError("targets are outside the logit class range.")
        temperatures = np.geomspace(0.05, 10.0, candidates)
        losses = []
        for temperature in temperatures:
            probabilities = np.apply_along_axis(lambda row: softmax(row / temperature), 1, values)
            losses.append(float(-np.log(np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)).mean()))
        return cls(float(temperatures[int(np.argmin(losses))]))


@dataclass(frozen=True)
class UnknownPolicy:
    """Validation-derived rules for returning Unknown with the best candidate."""

    min_router_probability: float
    min_joint_probability: float
    min_margin: float
    require_complete_routing: bool = True

    @classmethod
    def fit_known_coverage(
        cls,
        router_top_probabilities: Sequence[float],
        joint_probabilities: Sequence[float],
        margins: Sequence[float],
        target_known_coverage: float = 0.95,
        require_complete_routing: bool = True,
    ) -> "UnknownPolicy":
        """Set conservative lower thresholds from known validation examples.

        This controls rejection of known taxa. It does not claim open-set
        performance; an external non-orchid set is required for that later claim.
        """
        if not 0 < target_known_coverage <= 1:
            raise ValueError("target_known_coverage must be in (0, 1].")
        arrays = [np.asarray(values, dtype=float) for values in (router_top_probabilities, joint_probabilities, margins)]
        if any(values.ndim != 1 or values.size == 0 for values in arrays) or len({values.size for values in arrays}) != 1:
            raise ValueError("All validation score arrays must be non-empty and equally sized.")
        quantile = 1.0 - target_known_coverage
        return cls(*(float(np.quantile(values, quantile)) for values in arrays), require_complete_routing)


@dataclass(frozen=True)
class OpenSetDecision:
    is_unknown: bool
    reason: str | None
    best_candidate_species_id: str | None
    best_candidate_score: float | None

    @property
    def display_label(self) -> str:
        if not self.is_unknown:
            return self.best_candidate_species_id or "Unknown"
        if self.best_candidate_species_id is None:
            return "Unknown"
        return f"Unknown (best candidate: {self.best_candidate_species_id}, score: {self.best_candidate_score:.3f})"


def apply_unknown_policy(result: CascadeResult, policy: UnknownPolicy) -> OpenSetDecision:
    """Apply abstention after top-k routing while retaining the best candidate."""
    candidate = result.selected_species
    if candidate is None:
        return OpenSetDecision(True, result.fallback_reason or "no_species_candidate", None, None)
    if policy.require_complete_routing and result.fallback_reason:
        return OpenSetDecision(True, result.fallback_reason, candidate.species_id, candidate.joint_probability)
    top_router = result.routed_genera[0].router_probability
    runner_up = result.species_candidates[1].joint_probability if len(result.species_candidates) > 1 else 0.0
    margin = candidate.joint_probability - runner_up
    if top_router < policy.min_router_probability:
        return OpenSetDecision(True, "low_genus_confidence", candidate.species_id, candidate.joint_probability)
    if candidate.joint_probability < policy.min_joint_probability:
        return OpenSetDecision(True, "low_species_confidence", candidate.species_id, candidate.joint_probability)
    if margin < policy.min_margin:
        return OpenSetDecision(True, "low_candidate_margin", candidate.species_id, candidate.joint_probability)
    return OpenSetDecision(False, None, candidate.species_id, candidate.joint_probability)


@dataclass(frozen=True)
class HierarchicalSelectivePolicy:
    """Calibration-only decision rule for flat and dual-head orchid models."""

    species_temperature: float
    genus_temperature: float
    min_species_probability: float
    min_genus_probability: float
    min_species_margin: float
    target_known_coverage: float
    uses_genus_head: bool


def fit_hierarchical_selective_policy(
    species_logits: np.ndarray,
    species_targets: Sequence[int],
    taxonomy: OrchidTaxonomyIndex,
    genus_logits: np.ndarray | None = None,
    target_known_coverage: float = 0.95,
) -> HierarchicalSelectivePolicy:
    """Fit temperatures and abstention thresholds only from calibration rows."""
    if not 0 < target_known_coverage <= 1:
        raise ValueError("target_known_coverage must be in (0, 1].")
    species_scaler = TemperatureScaler.fit(species_logits, species_targets)
    species_probabilities = np.asarray([species_scaler.transform(row) for row in species_logits])
    species_top = species_probabilities.max(axis=1)
    sorted_species = np.sort(species_probabilities, axis=1)
    margin = sorted_species[:, -1] - sorted_species[:, -2]
    genus_targets = taxonomy.genus_targets(torch.as_tensor(species_targets, dtype=torch.long)).cpu().numpy()
    uses_genus_head = genus_logits is not None
    if genus_logits is not None:
        genus_scaler = TemperatureScaler.fit(genus_logits, genus_targets)
        genus_probabilities = np.asarray([genus_scaler.transform(row) for row in genus_logits])
    else:
        genus_scaler = TemperatureScaler(1.0)
        species_tensor = torch.as_tensor(species_logits, dtype=torch.float32)
        genus_probabilities = aggregate_species_probabilities(species_tensor / species_scaler.temperature, taxonomy).cpu().numpy()
    genus_top = genus_probabilities.max(axis=1)
    quantile = 1.0 - target_known_coverage
    return HierarchicalSelectivePolicy(
        species_temperature=species_scaler.temperature,
        genus_temperature=genus_scaler.temperature,
        min_species_probability=float(np.quantile(species_top, quantile)),
        min_genus_probability=float(np.quantile(genus_top, quantile)),
        min_species_margin=float(np.quantile(margin, quantile)),
        target_known_coverage=target_known_coverage,
        uses_genus_head=uses_genus_head,
    )


def hierarchical_decisions(
    species_logits: np.ndarray,
    taxonomy: OrchidTaxonomyIndex,
    policy: HierarchicalSelectivePolicy,
    genus_logits: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Return species, genus, or Unknown decisions without tuning on this data."""
    species_probabilities = np.asarray([softmax(row / policy.species_temperature) for row in species_logits])
    species_tensor = torch.as_tensor(species_logits / policy.species_temperature, dtype=torch.float32)
    implied_genus = aggregate_species_probabilities(species_tensor, taxonomy).cpu().numpy()
    if policy.uses_genus_head:
        if genus_logits is None:
            raise ValueError("This policy requires genus logits.")
        genus_probabilities = np.asarray([softmax(row / policy.genus_temperature) for row in genus_logits])
    else:
        genus_probabilities = implied_genus
    decisions: list[dict[str, object]] = []
    for species_values, genus_values in zip(species_probabilities, genus_probabilities):
        ranked = np.argsort(-species_values, kind="mergesort")
        species_index = int(ranked[0])
        margin = float(species_values[ranked[0]] - species_values[ranked[1]])
        genus_index = int(np.argmax(genus_values))
        species_probability = float(species_values[species_index])
        genus_probability = float(genus_values[genus_index])
        if species_probability >= policy.min_species_probability and margin >= policy.min_species_margin:
            decisions.append({"decision_level": "species", "species_index": species_index, "genus_index": taxonomy.species_to_genus[species_index], "confidence": species_probability, "margin": margin})
        elif genus_probability >= policy.min_genus_probability:
            decisions.append({"decision_level": "genus", "species_index": None, "genus_index": genus_index, "confidence": genus_probability, "margin": margin})
        else:
            decisions.append({"decision_level": "unknown", "species_index": None, "genus_index": None, "confidence": max(species_probability, genus_probability), "margin": margin})
    return decisions


def forced_species_decisions(
    species_logits: np.ndarray,
    taxonomy: OrchidTaxonomyIndex,
    policy: HierarchicalSelectivePolicy,
) -> list[dict[str, object]]:
    """Return calibrated forced-species decisions for non-HSC baselines."""
    probabilities = np.asarray([softmax(row / policy.species_temperature) for row in species_logits])
    decisions: list[dict[str, object]] = []
    for values in probabilities:
        ranked = np.argsort(-values, kind="mergesort")
        species_index = int(ranked[0])
        decisions.append(
            {
                "decision_level": "species",
                "species_index": species_index,
                "genus_index": taxonomy.species_to_genus[species_index],
                "confidence": float(values[species_index]),
                "margin": float(values[ranked[0]] - values[ranked[1]]),
            }
        )
    return decisions
