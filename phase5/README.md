# Phase 5 — Benchmarking, Ablations, and Interpretability

**Author:** Khalid Zaman, Research Assistant Professor (RAP) · **Supervisor:** Prof. Zhaoxi Sun, SUAT
**Project:** multimodal-ppi (Protein–Protein Interaction)

Phase 5 compares the project's models against published methods and explains their
predictions. It has three parts:

1. **Interaction benchmark** — our `CrossAttnPPI-650M` vs the published
   `PLM-interact-650M`, both scored on the identical leakage-free Bernett test set
   (52,048 pairs), same metrics (AUROC / AUPR). Our model is trained on the Bernett
   train split; the comparison model is used as released. Job: `bench_interaction.sbatch`.
2. **Ablation** — sequence-only → + structure → + evolution on the held-out
   PPB-Affinity test split (affinity + interface), completing the Phase 3 ablation.
3. **Interpretability** — cross-attention interaction maps and predicted interface
   residues shown on experimental 3D structures.

## Files

- `bench_interaction.sbatch` — runs both interaction models on the Bernett test set.
- `logs/` — SLURM job logs (kept out of git).

Large scored outputs are written to `~/Projects/ppi-data/phase5_bench/` (cluster only,
kept out of git). Final numbers and figures are recorded in `reports/PHASE5_SUMMARY.md`.
