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

## Reproducible experiment entry points

The checked-in YAML files define the baseline and router protocols. Edit only the
private `root_dir` and manifest location if your local paths differ; keep a copy of
the final YAML beside the resulting artifact directory.

```powershell
conda run -n orchid_edge python scripts/train_orchid_baseline.py
conda run -n orchid_edge python scripts/train_orchid_router.py
conda run -n orchid_edge python scripts/train_orchid_experts.py --genus Dendrobium
```

`--genus all` deliberately starts every available expert and should be used only
after the router and a representative expert have been checked. Each run writes
label and configuration provenance under `artifacts/orchid/<experiment-id>/`.

## Checkpoint bundles

Every launcher writes its best validation-loss model to
`checkpoints/best_orchid_model.pt`. This is a versioned PyTorch bundle, not a raw
`state_dict`; it stores the class-label order, task, optional target genus, model
name, input size, ImageNet normalization, split-manifest path, and taxonomy file
reference with the weights. Export and routing code must load this bundle through
`load_orchid_checkpoint`, so a router cannot accidentally be paired with an expert
whose label order or preprocessing differs.

Evaluate only the held-out manifest test split with explicit, self-describing model
bundles. The command writes `metrics.json` and image-level `predictions.csv`; it
does not evaluate on train or validation images.

```powershell
conda run -n orchid_edge python scripts/run_orchid_evaluation.py `
  --router-checkpoint artifacts/orchid/genus_router/mobilenet_v2/checkpoints/best_orchid_model.pt `
  --expert-checkpoint Dendrobium=artifacts/orchid/species_experts/mobilenet_v2/Dendrobium/checkpoints/best_orchid_model.pt
```

Add `--unknown-policy path/to/policy.json` only after fitting thresholds on the
validation split. The reported metrics include router top-1/top-2 genus accuracy,
cascade species accuracy, known coverage, selective species accuracy, Unknown rate,
and taxonomic error cost. Phylogenetic error is added only after the reviewed mapping
reaches the configured coverage gate.

When coverage is ready, add `--phylogeny-tree-directory` and
`--phylogeny-mapping`; the evaluator will use the posterior-mean distance matrix
and include `mean_phylogenetic_error` in `metrics.json`.

## LiteRT export and compressed model packs

Export each finalized checkpoint with a distinct filename, then merge the emitted
entry manifests and pack them. A pack is ZIP-deflated for compressed-at-rest storage;
on installation it is safely extracted into a cache and every model checksum is
verified before use.

```powershell
conda run -n orchid_edge python scripts/export_orchid_litert.py `
  --checkpoint artifacts/orchid/genus_router/mobilenet_v2/checkpoints/best_orchid_model.pt `
  --output artifacts/orchid/genus_router/mobilenet_v2/exports/router.tflite `
  --role router --entry-output artifacts/orchid/genus_router/mobilenet_v2/exports/router_entry.json
```

Run the same command for each expert with `--role expert --genus <Genus>`. Then use
`build_orchid_deployment_manifest.py` followed by `package_orchid_models.py` to
create one verified pack. Export intentionally does not silently quantize models:
each quantization setting must be evaluated as its own experiment before deployment.

## Cascade decision contract

`HierarchicalCascadeRouter` receives router logits and the logits returned by the
available selected genus experts. It deterministically selects the top one or top
two genera and scores each candidate species using:

```text
P(genus | image) × P(species | selected genus, image)
```

The result retains both the unnormalized `joint_probability` and a
`fused_probability` normalized over the experts actually evaluated. If a selected
expert is absent, routing continues with the available selected experts and records
`missing_expert:<genus>`; if none are available it returns no species prediction.
Phase 6 will convert low-confidence or incomplete-routing outcomes into the
user-facing **Unknown** decision using validation-set calibration.

## Calibration and Unknown

Phase 6 calibrates a scalar temperature on a held-out validation set, then derives
the router, joint-species, and candidate-margin thresholds from a chosen known-taxon
coverage target. It returns **Unknown (best candidate: …, score: …)**, matching the
app's intended UI. These thresholds control abstention on known taxa only. Do not
claim non-orchid open-set performance until a separately held-out non-orchid set is
evaluated.

## Phylogenetic error cost

The optional source is Pérez-Escobar et al. (2024), *The origin and speciation of
orchids*, New Phytologist, DOI `10.1111/nph.19580`. Its CC-BY 4.0 Figshare record,
DOI `10.6084/m9.figshare.22245940.v1`, provides the 10 posterior species trees used
by this pipeline. The downloaded source archive is intentionally under ignored
`data/` and is checksum-verified against the published MD5 before use.

```powershell
conda run -n orchid_edge python scripts/prepare_orchid_phylogeny.py `
  --dataset-root data/taxonomic-orchid/TaxonomicOrchidDataset
```

This writes a review-required mapping CSV. Fill `source_tip` only after checking
accepted scientific names and set `mapping_status` to `matched`; cultivars, hybrids,
synonyms, and unresolved labels stay unmatched. The phylogenetic metric refuses to
run below 90% coverage by default. It reports mean normalized patristic error as a
secondary safety analysis, never as a substitute for top-1 accuracy.

All ten released posterior trees are used: the pipeline reports the posterior mean
distance matrix and its standard deviation rather than choosing a single topology.
The pinned archive, checksum, license, and aggregation rule are recorded in
`configs/orchid/phylogeny_source.yaml`.
