# Phase 5 — Benchmarking, Ablations, and Interpretability

**Author:** Khalid Zaman, Research Assistant Professor (RAP)   **Supervisor:** Prof. Zhaoxi Sun, SUAT
**Project:** multimodal-ppi (Protein–Protein Interaction)   ·   2026-10-09
**Status:** IN PROGRESS — interaction benchmark complete; module ablation complete (result honest and inconclusive at 650M); stratified analysis and fusion study in progress; interpretability remaining.

---

## Goal of Phase 5

Compare the project's models against published methods on shared, held-out data, test the central
hypothesis by ablation, and explain the predictions. Parts: (A) interaction benchmark against
PLM-interact, (B) module ablation sequence → + structure → + evolution, (B2) stratified analysis,
(B3) fusion study, (C) interpretability.

## Part A — Interaction benchmark (leakage-free Bernett test set)

Both models scored on the **identical** Bernett test set (52,048 pairs), same metrics, same 650M
backbone. Inference only.

| Model (same test set) | AUROC ↑ | AUPR ↑ |
|---|---|---|
| **This work — CrossAttnPPI-650M** | **0.717** | **0.709** |
| PLM-interact-650M (as released) | 0.622 | 0.636 |
| Published baselines (Bernett) | — | ~0.69 |

Our model (trained on the Bernett train split) leads PLM-interact (used as released) by ~+0.095
AUROC / +0.073 AUPR, and sits above the ~0.69 published baselines. Figure:
`reports/figures/phase5_interaction_benchmark.png`. Job: `phase5/bench_interaction.sbatch`.

## Part B — Module ablation (650M, held-out test set)

### Method

The three views were ablated at the definitive 650M scale. Each condition was trained with **3
seeds** and scored on the **held-out test split (757 complexes)**. The switches `--no_struct` /
`--no_evo` in `train_phase3.py` zero the structure and evolution inputs, so the conditions differ
only in which views are active. The full condition reuses the definitive runs (`mm_650M_s0/1/2`).
Job: `phase5/ablation_650M.sbatch`; aggregation: `phase5/agg_ablation.py`.

### Results — held-out test set (mean ± s.d., 3 seeds)

| Condition | Affinity Pearson ↑ | Affinity RMSE ↓ | Interface AUPR ↑ |
|---|---|---|---|
| Sequence only | 0.429 ± 0.026 | 1.593 ± 0.123 | 0.286 ± 0.007 |
| + Structure | 0.441 ± 0.014 | 1.592 ± 0.008 | 0.289 ± 0.010 |
| + Evolution (full) | 0.439 ± 0.018 | 1.575 ± 0.044 | 0.281 ± 0.006 |

Figure: `reports/figures/phase5_ablation.png`.

### Findings (honest)

1. **At 650M on the held-out test set, the three conditions are statistically tied** — every
   pairwise difference is small and falls within the error bars. We therefore **do not claim** that
   structure or evolution significantly improve affinity or interface prediction at this scale on
   this test set.
2. **This refines the earlier Phase 3 ablation.** That 35M comparison (single run, validation split,
   no error bars) suggested evolution helped (Pearson 0.293 → 0.343). The rigorous version here
   (3 seeds, test split, error bars) does **not** confirm it — the earlier signal was within noise.
   Reporting this plainly is the correct scientific response, and is exactly why the rigorous
   version was run.
3. **Likely reasons:** the 650M sequence encoder is already strong (diminishing returns from extra
   views); the structure view aligns for only ~60% of complexes (diluting its signal across the full
   test set); and the 757-complex test set limits the resolution of small effects.

### Part B2 — Stratified analysis (in progress)

Because structure is available for only ~60% of complexes and MSA depth varies, the full-test average
may hide a real effect. We therefore measure the **structure** benefit on the subset of test
complexes that have an aligned structure, and the **evolution** benefit on deep-MSA complexes —
asking whether each view helps *where it applies*. The result is reported regardless of outcome.

### Part B3 — Fusion study (in progress)

The current fusion is additive (`h = seq + structure + evolution`). We test whether a stronger
fusion (e.g. gated or attention-based) recovers a benefit from the extra views.

## Part C — Interpretability (pending)

Cross-attention interaction maps and predicted interface residues on experimental 3D structures.

## Honest overall position

The robust, defensible results do **not** depend on the ablation: (i) the interaction model clearly
beats the published PLM-interact on the leakage-free benchmark, and (ii) the project delivers a
complete, reproducible multimodal pipeline with solid affinity prediction and honest error bars. The
one claim we cannot currently support is that structure + evolution significantly improve affinity /
interface prediction at 650M; this is reported honestly and investigated in Parts B2 and B3.

## Files

- `phase5/bench_interaction.sbatch`, `phase5/ablation_650M.sbatch`, `phase5/ablation_smoke.sbatch`,
  `phase5/agg_ablation.py` — benchmark and ablation jobs.
- `reports/figures/phase5_interaction_benchmark.png/.pdf`, `reports/figures/phase5_ablation.png/.pdf`.
- View switches added to `phase3/train_phase3.py` (`--no_struct`, `--no_evo`) and `phase3/data.py`.

## Next steps

1. Part B2 — stratified analysis (structure on structure-available subset; evolution on deep-MSA).
2. Part B3 — stronger fusion, retrained and evaluated the same way.
3. Part C — interpretability figures.
4. (Optional) a small inference script: two sequences in → interaction / affinity / interface out.
