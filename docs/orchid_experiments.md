# Orchid experiment protocol

This is the execution order for results that can support the WACV Applications
Track paper. Do not report a number before the corresponding checklist item is
complete.

## 1. Freeze the data protocol

1. Snapshot the private `Genus/Species/image` hierarchy.
2. Create the split manifest with `tools/orchid_split_audit.py create-manifest`.
3. Run the leakage audit and manually review every cross-split candidate.
4. Record known limitations: the initial split is image-stratified, not proven
   specimen-disjoint, because repeated captures lack specimen IDs.
5. Never move images after the reported manifest is frozen.

## 2. Train the required comparisons

Use the same manifest, backbone family, image preprocessing, epoch budget, and
reporting split for every comparison.

| ID | Model | Required result |
| --- | --- | --- |
| B0 | `flat_species` | 199-way single-model baseline |
| B1 | `genus` | router top-1 and top-2 genus accuracy |
| H1 | top-1 router + available expert | end-to-end species accuracy |
| H2 | top-2 router + probability fusion | end-to-end species accuracy and cost |
| H3 | H2 + calibrated Unknown | coverage, selective accuracy, Unknown rate |

Run `scripts/train_orchid_baseline.py`, `scripts/train_orchid_router.py`, and
`scripts/train_orchid_experts.py --genus <Genus>`. Use `--genus all` only after a
single representative expert has completed and its labels have been checked.

A genus with exactly one represented species is a deterministic specialist: its
conditional species probability is 1.0 after a genus decision. Do not present its
zero-loss one-class training run as learned evidence; report it as deterministic
routing and implement it as such in the mobile pack.

## 3. Calibrate and evaluate

Fit temperature and Unknown thresholds only on validation predictions. Freeze them
before calling `scripts/run_orchid_evaluation.py`, which uses the held-out test split.
Report the resulting `metrics.json` and preserve `predictions.csv` for error review.

Do not use test metrics to select the top-k policy, confidence thresholds, epochs,
or model architecture. If choices change, create a new experiment ID and repeat the
test evaluation once.

## 4. Phylogenetic analysis

Review the generated mapping CSV with accepted names. Run the optional phylogenetic
metric only if the configured coverage threshold is met, and disclose its coverage
and the count of excluded labels. It is an error-severity analysis, not evidence that
the visual model learned evolutionary relationships.

## 5. Edge deployment

Export only the frozen winning router and experts to LiteRT. Record the original
PyTorch and LiteRT results, model size, device latency, peak memory where available,
and Android version/device model. Package models only after their checksums and
deployment manifest validate.

## 6. Before writing results

- All tables derive from held-out predictions and include sample counts.
- Report class imbalance and species with no images separately.
- State image ownership, label-review process, and non-release policy.
- State that no image, location, or inference data leave the device.
- Use an external non-orchid set before claiming open-set or Unknown performance.
- Run `conda run -n orchid_edge python -m unittest discover -s tests -v`.
