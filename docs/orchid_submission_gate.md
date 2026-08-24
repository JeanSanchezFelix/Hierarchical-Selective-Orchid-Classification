# Orchid Paper Submission Gate

Run the following only after the public dataset manifest is frozen and every
condition has completed for seeds `17`, `42`, and `123`:

```powershell
python scripts/validate_orchid_submission.py `
  --dataset-manifest data/orchidaceae-inat-v1/manifests/split.csv
```

The command fails if any condition lacks held-out metrics or image-level
predictions, or if the hAURC summary and paired-bootstrap artifacts are absent.
It is a reproducibility gate, not evidence that a paper has passed peer review.

Before submitting, additionally verify:

- realized dataset counts replace all approximate counts in the manuscript;
- the generated WACV results table and risk-coverage figure match the frozen
  artifacts;
- the paper reports host-only edge measurements as host-only;
- any LiteRT or INT8 claim has a separate, matching parity report;
- no non-orchid or physical-device claim appears without its evaluation data.
