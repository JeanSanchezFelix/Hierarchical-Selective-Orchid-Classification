# Orchid Research Cleanup Manifest

## Status

Phase 1 inventory only. This file is not deletion authorization. No listed
path may be removed, moved, or rewritten until the user approves the relevant
cleanup action in a later phase.

## Keep And Refactor

| Path | Reason | Phase |
| --- | --- | --- |
| `datasets/TaxonomicOrchidDataset.py` | Existing taxonomy-aware folder scanner; refactor for the public data contract | 2-3 |
| `datasets/CustomHierarchicalDataset.py` | Existing hierarchy and image-folder support; inspect before replacement | 2-3 |
| `model_compression/src/orchid/` | Orchid-specific taxonomy, calibration, export, artifact, and evaluation foundations | 3-4 |
| `scripts/train_orchid_*.py` | Cascade baselines and migration reference | 3 |
| `scripts/run_orchid_evaluation.py` | Cascade baseline evaluation reference | 3-4 |
| `scripts/prepare_orchid_phylogeny.py` | Optional error-severity analysis | 4 |
| `tools/orchid_split_audit.py` | Duplicate and split auditing base | 2 |
| `configs/orchid/` | Existing configuration location | 2-5 |
| `tests/test_orchid_*.py` | Orchid contract test base | 2-5 |
| `docs/orchid_*.md` | Existing documentation to be superseded or updated after the new harness exists | 6 |
| `requirements.txt` | Dependency source; narrow only after the final import audit | 6 |

## Replace After Parity Tests

| Path | Reason | Replacement condition |
| --- | --- | --- |
| `scripts/orchid_training.py` | Task-specific single-head launcher cannot express all proposed methods | Unified configuration runner passes smoke and baseline parity tests |
| `model_compression/src/orchid/evaluation.py` | Current report shape is cascade-centered | New evaluator reproduces C1/C2 reports and emits common prediction schema |
| `model_compression/src/orchid/calibration.py` | Current policy is cascade-specific | Hierarchical calibration unit and integration tests pass |
| `docs/orchid_experiments.md` | Current protocol centers the cascade and Android deployment | New paper protocol and setup guide are verified |
| `docs/orchid_setup.md` | Current guide assumes a private dataset and cascade workflow | Public acquisition and unified runner are verified |

## Candidate Removal After Approval

| Path or area | Reason | Required check before removal |
| --- | --- | --- |
| `datasets/CpAnemiaDataset.py` | Unrelated biomedical dataset | No import from retained orchid modules |
| `datasets/MonkeypoxDataset.py` | Unrelated biomedical dataset | No import from retained orchid modules |
| `datasets/SkinCancerDataset.py` | Unrelated biomedical dataset | No import from retained orchid modules |
| `notebooks/` | Unrelated or obsolete exploratory notebooks | Preserve only a named orchid analysis notebook if it remains necessary |
| `model_compression/src/tensorrt/` | Not part of the LiteRT edge-ready study | No retained imports or documentation references |
| Generic biomedical sections of `main.py`, `knowledge_distillation.py`, and `pytest.py` | Legacy entry points outside the orchid paper surface | Unified orchid entry point and test command pass |
| Generic converter and quantization modules not imported by LiteRT export | Remove only after import graph and export tests | `rg` import audit plus export smoke test |
| Existing private `data/taxonomic-orchid/` | Not the paper dataset | Explicit user confirmation; never delete as part of automated cleanup |

## Explicitly Protected

- The staged `README.md` change belongs to the user and is not part of this
  cleanup manifest.
- Git history, generated artifacts, and private data are never removed by a
  cleanup command without an explicit target and user approval.
- The Android project is outside this repository cleanup scope.

## Approval Gate

Before Phase 6 cleanup, the user must approve this manifest or a revised
version. Deletion then occurs in small, verified groups, with an import audit
and tests after each group.
