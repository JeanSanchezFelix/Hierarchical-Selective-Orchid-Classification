"""Validation-set calibration and abstention rules for the orchid cascade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .routing import CascadeResult, softmax


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
