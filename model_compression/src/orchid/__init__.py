"""Orchid-specific contracts used by the hierarchical classifier pipeline."""

from .artifacts import OrchidArtifactLayout
from .checkpoints import OrchidModelCheckpoint, load_orchid_checkpoint, save_orchid_checkpoint
from .routing import HierarchicalCascadeRouter, validate_cascade_metadata
from .calibration import TemperatureScaler, UnknownPolicy, apply_unknown_policy
from .phylogeny import PEREZ_ESCOBAR_2024, build_distance_matrix, build_posterior_distance_summary, mean_phylogenetic_error
from .evaluation import evaluate_cascade, summarize_records, write_evaluation_report
from .deployment_manifest import make_deployment_manifest, deployment_entry
from .model_packs import create_model_pack, install_model_pack
from .taxonomy import OrchidTaxonomy, TaxonRecord, scan_orchid_taxonomy

__all__ = [
    "OrchidArtifactLayout",
    "OrchidModelCheckpoint",
    "HierarchicalCascadeRouter",
    "TemperatureScaler",
    "UnknownPolicy",
    "apply_unknown_policy",
    "PEREZ_ESCOBAR_2024",
    "build_distance_matrix",
    "build_posterior_distance_summary",
    "mean_phylogenetic_error",
    "evaluate_cascade",
    "summarize_records",
    "write_evaluation_report",
    "make_deployment_manifest",
    "deployment_entry",
    "create_model_pack",
    "install_model_pack",
    "OrchidTaxonomy",
    "TaxonRecord",
    "scan_orchid_taxonomy",
    "load_orchid_checkpoint",
    "save_orchid_checkpoint",
    "validate_cascade_metadata",
]
