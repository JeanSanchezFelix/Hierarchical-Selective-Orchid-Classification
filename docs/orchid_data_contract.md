# Orchid data and artifact contract

The private image collection is expected to use this layout:

```text
<dataset-root>/<Genus>/<Species>/<image-file>
```

`UNLABELED` folders are excluded at either taxonomy level. The stable species
identifier is `Genus::Species`; do not rely on a bare species folder name because
the same epithet can occur under different genera.

## Dataset tasks

| Task | Purpose | Class labels | Required option |
| --- | --- | --- | --- |
| `genus` | First-stage router | genus names | none |
| `flat_species` | Single-model baseline | `Genus::Species` | none |
| `genus_species` | Genus-specific species expert | `Genus::Species` within one genus | `target_genus` |
| `target_genus` | Compatibility alias for a genus-specific expert | `Genus::Species` within one genus | `target_genus` |

Use a reviewed split manifest for every reported experiment. A full manifest may
be reused for a genus expert: rows outside its target genus are ignored, but every
image included by that expert must appear exactly once in the manifest.

```python
from model_compression.src.utils.preprocessing import load_data

loaders = load_data(
    "TaxonomicOrchid",
    split_manifest="artifacts/leakage_audit/orchid_split.csv",
    dataset_kwargs={
        "task": "genus_species",
        "target_genus": "Dendrobium",
    },
)
```

## Artifacts

Future train and export scripts create an `OrchidArtifactLayout` under a named
experiment directory. It reserves `checkpoints/`, `exports/`, and `reports/`, plus
`taxonomy.json` and `run_metadata.json`. These files contain labels and provenance,
not private image data.

The phylogenetic source, label mapping, unmatched labels, and derived distance
matrix will be added in Phase 6. Phylogenetic metrics will only be calculated for
labels that map to a vetted source tree.
