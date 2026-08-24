# Orchid HSC Research Deliverable

This repository's supported research surface is the public Orchidaceae
benchmark and the taxonomy-consistent hierarchical selective classification
comparison described in `docs/orchid_paper_protocol.md`.

Start here:

1. Build a manifest with `scripts/prepare_public_orchid_dataset.py`.
2. Run each single-model condition with `scripts/run_orchid_experiment.py`.
3. Run cascade controls with `scripts/run_orchid_cascade.py`.
4. Summarize all three seeds with `scripts/summarize_orchid_paper_results.py`.
5. Audit frozen checkpoints using `scripts/audit_orchid_edge.py`.

Private images, generated artifacts, and Android deployment are intentionally
outside this deliverable and are not required to reproduce the public study.
