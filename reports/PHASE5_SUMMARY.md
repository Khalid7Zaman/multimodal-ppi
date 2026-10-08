# Phase 5 — Benchmarking, Ablations, and Interpretability

**Author:** Khalid Zaman, Research Assistant Professor (RAP)   **Supervisor:** Prof. Zhaoxi Sun, SUAT
**Project:** multimodal-ppi (Protein–Protein Interaction)   ·   2026-10-08
**Status:** IN PROGRESS — interaction benchmark complete; ablation and interpretability remaining.

---

## Goal of Phase 5

Compare the project's models against published methods on shared, held-out data, and explain
their predictions. Phase 5 has three parts: (A) an interaction benchmark against the published
PLM-interact model, (B) the sequence → + structure → + evolution ablation on the affinity /
interface tasks, and (C) interpretability figures. This document records part A; parts B and C
are in progress.

## Part A — Interaction benchmark (leakage-free Bernett test set)

### Method

Both models are scored on the **identical** Bernett gold-standard test set (**52,048 pairs**,
balanced, leakage-free) using the **same two metrics** (AUROC and AUPR). Both use a 650M ESM-2
backbone. This is an inference-only comparison; no retraining.

- **This work (`CrossAttnPPI-650M`)** — trained on the Bernett train split (the leakage-free
  setting the benchmark is designed for), evaluated on the Bernett test split.
- **PLM-interact-650M** — the published model used **as released** (trained by its authors on
  their own human-PPI data, not re-tuned for this benchmark).

Job: `phase5/bench_interaction.sbatch`; raw scores under `~/Projects/ppi-data/phase5_bench/`.

### Results — held-out test set (52,048 pairs)

| Model (same test set) | AUROC ↑ | AUPR ↑ |
|---|---|---|
| **This work — CrossAttnPPI-650M** | **0.717** | **0.709** |
| PLM-interact-650M (as released) | 0.622 | 0.636 |
| Published baselines (Bernett) | — | ~0.69 |

Figure: `reports/figures/phase5_interaction_benchmark.png`.

### Findings

1. **Our model leads on the leakage-free benchmark.** CrossAttnPPI-650M reaches AUROC 0.717 /
   AUPR 0.709, ahead of PLM-interact used as released (0.622 / 0.636) by about +0.095 AUROC and
   +0.073 AUPR, and above the ~0.69 published AUPR baselines on this benchmark.
2. **Honest framing.** Our model was trained on the Bernett train split, while PLM-interact was
   applied as released (not re-tuned for this split). Part of the gap therefore reflects
   task-matched training, not architecture alone. A fully controlled architecture-vs-architecture
   comparison — fine-tuning PLM-interact on the Bernett train split — is noted as optional further
   work.
3. **This supports the project's premise.** The large drop of an as-released sequence model on the
   leakage-free test (to AUROC 0.62) illustrates the generalisation gap the proposal set out to
   address: sequence-only models transfer poorly to proteins outside their training distribution.
   A task-matched model handles it better.

## Part B — Ablation (in progress)

Complete the sequence-only → + structure → + evolution ablation on the held-out PPB-Affinity test
split (affinity + interface), extending the Phase 3 ablation so each module's contribution is shown
on one fixed split.

## Part C — Interpretability (in progress)

Cross-attention interaction maps (which residues each protein attends to) and predicted interface
residues shown on experimental 3D structures.

## Files

- `phase5/bench_interaction.sbatch` — the interaction benchmark job (both models, same test set).
- `phase5/README.md` — Phase 5 overview.
- `reports/figures/phase5_interaction_benchmark.png/.pdf` — the benchmark figure.

Large scored outputs stay on the cluster under `~/Projects/ppi-data/phase5_bench/` (out of git).
Code and documentation are in git, mirrored to Gitee and GitHub.

## Next steps

1. Ablation on the test split (Part B).
2. Interpretability figures (Part C).
3. (Optional) a small inference script — two sequences in → interaction / affinity / interface out.
4. (Optional, for maximal rigour) fine-tune PLM-interact on the Bernett train split for a fully
   controlled architecture comparison.
