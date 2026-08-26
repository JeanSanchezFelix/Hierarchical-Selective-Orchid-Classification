"""Edge-compatible MobileNetV2 models for the orchid comparison matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision import models


METHOD_CASCADE_TOP1 = "cascade_top1"
METHOD_CASCADE_TOP2 = "cascade_top2"
METHOD_FLAT_CE = "flat_ce"
METHOD_FLAT_BALANCED_SOFTMAX = "flat_balanced_softmax"
METHOD_FLAT_HSC = "flat_hsc"
METHOD_DUAL_HEAD = "dual_head"
METHOD_DUAL_HEAD_HSC = "dual_head_taxonomy_hsc"

CASCADE_METHODS = frozenset({METHOD_CASCADE_TOP1, METHOD_CASCADE_TOP2})
FLAT_METHODS = frozenset({METHOD_FLAT_CE, METHOD_FLAT_BALANCED_SOFTMAX, METHOD_FLAT_HSC})
DUAL_HEAD_METHODS = frozenset({METHOD_DUAL_HEAD, METHOD_DUAL_HEAD_HSC})
HSC_METHODS = frozenset({METHOD_FLAT_HSC, METHOD_DUAL_HEAD_HSC})
TRAINABLE_METHODS = FLAT_METHODS | DUAL_HEAD_METHODS
ALL_METHODS = CASCADE_METHODS | TRAINABLE_METHODS


@dataclass(frozen=True)
class OrchidTaxonomyIndex:
    """Stable species-to-genus mapping shared by model, loss, and evaluation."""

    species_ids: tuple[str, ...]
    genus_ids: tuple[str, ...]
    species_to_genus: tuple[int, ...]

    @classmethod
    def from_species_ids(cls, species_ids: Sequence[str]) -> "OrchidTaxonomyIndex":
        labels = tuple(species_ids)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("species_ids must be non-empty and unique.")
        genera: list[str] = []
        mapping: list[int] = []
        for species_id in labels:
            if "::" not in species_id:
                raise ValueError(f"Expected qualified species ID 'Genus::species', got {species_id!r}.")
            genus, species = species_id.split("::", 1)
            if not genus or not species:
                raise ValueError(f"Invalid qualified species ID: {species_id!r}.")
            if genus not in genera:
                genera.append(genus)
            mapping.append(genera.index(genus))
        return cls(labels, tuple(genera), tuple(mapping))

    def genus_targets(self, species_targets: torch.Tensor) -> torch.Tensor:
        index = torch.tensor(self.species_to_genus, device=species_targets.device, dtype=torch.long)
        if species_targets.numel() and (species_targets.min() < 0 or species_targets.max() >= len(self.species_ids)):
            raise ValueError("species target is outside the taxonomy label range.")
        return index[species_targets]


class MobileNetV2FeatureExtractor(nn.Module):
    """MobileNetV2 feature extractor with the classifier removed exactly once."""

    def __init__(self, use_imagenet_weights: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if use_imagenet_weights else None
        model = models.mobilenet_v2(weights=weights)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.feature_dim = model.last_channel

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = self.features(images)
        values = self.pool(values)
        values = torch.flatten(values, 1)
        return self.dropout(values)


class FlatSpeciesMobileNetV2(nn.Module):
    """One edge-compatible MobileNetV2 classifier for forced or post-hoc HSC labels."""

    def __init__(self, num_species: int, use_imagenet_weights: bool = True) -> None:
        super().__init__()
        if num_species < 2:
            raise ValueError("A flat species classifier requires at least two species.")
        self.backbone = MobileNetV2FeatureExtractor(use_imagenet_weights=use_imagenet_weights)
        self.species_head = nn.Linear(self.backbone.feature_dim, num_species)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.species_head(self.backbone(images))


class DualHeadMobileNetV2(nn.Module):
    """One MobileNetV2 backbone with species and genus heads in one forward pass."""

    def __init__(self, num_species: int, num_genera: int, use_imagenet_weights: bool = True) -> None:
        super().__init__()
        if num_species < 2 or num_genera < 2:
            raise ValueError("Dual-head taxonomy classification requires at least two species and two genera.")
        self.backbone = MobileNetV2FeatureExtractor(use_imagenet_weights=use_imagenet_weights)
        self.species_head = nn.Linear(self.backbone.feature_dim, num_species)
        self.genus_head = nn.Linear(self.backbone.feature_dim, num_genera)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        return self.species_head(features), self.genus_head(features)


def build_orchid_model(method: str, taxonomy: OrchidTaxonomyIndex, use_imagenet_weights: bool = True) -> nn.Module:
    """Build exactly one trainable model from the frozen comparison matrix."""
    if method in FLAT_METHODS:
        return FlatSpeciesMobileNetV2(len(taxonomy.species_ids), use_imagenet_weights=use_imagenet_weights)
    if method in DUAL_HEAD_METHODS:
        return DualHeadMobileNetV2(
            len(taxonomy.species_ids), len(taxonomy.genus_ids), use_imagenet_weights=use_imagenet_weights
        )
    if method in CASCADE_METHODS:
        raise ValueError(f"{method} is a legacy multi-model baseline; construct its router and experts through the cascade launcher.")
    raise ValueError(f"Unknown orchid method {method!r}. Allowed methods: {sorted(ALL_METHODS)}")


def aggregate_species_probabilities(species_logits: torch.Tensor, taxonomy: OrchidTaxonomyIndex) -> torch.Tensor:
    """Marginalize leaf probabilities into genus probabilities."""
    if species_logits.ndim != 2 or species_logits.shape[1] != len(taxonomy.species_ids):
        raise ValueError("species_logits shape does not match the taxonomy species count.")
    probabilities = torch.softmax(species_logits, dim=1)
    indices = torch.tensor(taxonomy.species_to_genus, device=species_logits.device, dtype=torch.long)
    result = torch.zeros(
        (species_logits.shape[0], len(taxonomy.genus_ids)), device=species_logits.device, dtype=species_logits.dtype
    )
    return result.index_add(1, indices, probabilities)


class BalancedSoftmaxLoss(nn.Module):
    """Long-tail cross-entropy using declared training class counts."""

    def __init__(self, class_counts: Sequence[int | float]) -> None:
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or counts.numel() < 2 or torch.any(counts <= 0):
            raise ValueError("Balanced Softmax requires one positive count for each class.")
        self.register_buffer("log_class_counts", torch.log(counts))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.log_class_counts.numel():
            raise ValueError("Logits shape does not match the Balanced Softmax class counts.")
        return functional.cross_entropy(logits + self.log_class_counts, targets)


class OrchidMultiTaskLoss(nn.Module):
    """Species, genus, and optional taxonomy-consistency objectives."""

    def __init__(
        self,
        taxonomy: OrchidTaxonomyIndex,
        species_loss: nn.Module,
        genus_weight: float = 1.0,
        consistency_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if genus_weight < 0 or consistency_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        self.taxonomy = taxonomy
        self.species_loss = species_loss
        self.genus_weight = genus_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        species_logits: torch.Tensor,
        species_targets: torch.Tensor,
        genus_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        species = self.species_loss(species_logits, species_targets)
        components = {"species_loss": species.detach()}
        total = species
        if genus_logits is not None:
            genus_targets = self.taxonomy.genus_targets(species_targets)
            genus = functional.cross_entropy(genus_logits, genus_targets)
            total = total + self.genus_weight * genus
            components["genus_loss"] = genus.detach()
            if self.consistency_weight:
                aggregated = aggregate_species_probabilities(species_logits, self.taxonomy).clamp_min(1e-12)
                consistency = functional.kl_div(functional.log_softmax(genus_logits, dim=1), aggregated, reduction="batchmean")
                total = total + self.consistency_weight * consistency
                components["consistency_loss"] = consistency.detach()
        components["total_loss"] = total.detach()
        return total, components
