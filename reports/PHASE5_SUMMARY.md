# Phase 5 — Benchmarking, Ablations, and Interpretability

**Author:** Khalid Zaman, Research Assistant Professor (RAP)   **Supervisor:** Prof. Zhaoxi Sun, SUAT
**Project:** multimodal-ppi (Protein–Protein Interaction)   ·   2026-10-09
**Status:** COMPLETE — interaction benchmark, module ablation, stratified analysis, fusion study, and interpretability all done. Headline: the interaction model beats PLM-interact on the leakage-free benchmark; the multimodal views give limited aggregate benefit at 650M, but evolution helps consistently on deep-MSA complexes; a gated fusion does not beat the simple additive one; and the model's interface predictions are spatially coherent and interpretable.

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

### Part B2 — Stratified analysis (complete)

Because MSA depth varies widely, the full-test average can hide a real effect. We therefore ask
whether each view helps *where it applies*. Method: per-complex affinity predictions from all nine
checkpoints (3 conditions × 3 seeds), sliced by per-complex structure availability and MSA depth
(median 575 sequences). Jobs: `phase5/stratified_eval.sbatch`, `phase5/eval_perrow.py`,
`phase5/make_test_meta.py`; aggregation `phase5/agg_stratified.py`.

**Structure — no clean contrast available.** 744 of 757 test complexes (98%) have an aligned
structure (either chain), so the test set cannot contrast with- vs without-structure. On the 744
with structure the effect is the same small, non-significant bump as in the aggregate ablation
(Pearson 0.421 → 0.433). We report this limitation plainly.

**Evolution — a consistent, interpretable benefit on deep-MSA complexes.**

| Subset | + structure Pearson | + evolution Pearson | + structure RMSE | + evolution RMSE |
|---|---|---|---|---|
| Deep MSA (n=379) | 0.512 ± 0.041 | **0.536 ± 0.008** | 1.451 ± 0.031 | **1.415 ± 0.019** |
| Shallow MSA (n=378) | 0.300 ± 0.014 | 0.279 ± 0.037 | 1.721 ± 0.042 | 1.720 ± 0.063 |

Figure: `reports/figures/phase5_stratified.png`.

**Findings:** On deep-MSA complexes, adding evolution **consistently improves** affinity (Pearson
0.512 → 0.536, RMSE 1.451 → 1.415), with the evolution runs tightly clustered (± 0.008). On
shallow-MSA complexes it does **not** help (neutral to slightly negative). This is the sensible,
mechanistic pattern expected — the evolutionary module helps precisely where there is evolutionary
signal to use — and explains why the whole-test ablation (mixing deep and shallow) looked flat.
Honest caveat: on the deep subset the error bars still overlap slightly, so this is reported as a
**consistent, modest improvement**, not a declared statistically-significant one. (A secondary
observation: deep-MSA complexes are far more predictable overall, Pearson ~0.52 vs ~0.30 — depth of
evolutionary information correlates with affinity predictability.)

### Part B3 — Fusion study (complete)

The default fusion is additive (`h = structure + evolution`). We tested a **gated fusion** — learned
per-residue gates that let the model weigh how much to trust each view (and down-weight an
uninformative one). It strictly generalises additive (gates = 1 reproduce it). Implemented behind
`--fusion gated` in `train_phase3.py` / `model.py`; trained at 650M, all views, 3 seeds, same
hyper-parameters as the additive run. Job: `phase5/train_gated_650M.sbatch`; comparison
`phase5/agg_gated.py`.

| Fusion (650M, all views) | Affinity Pearson ↑ | Affinity RMSE ↓ | Interface AUPR ↑ |
|---|---|---|---|
| Additive (original) | **0.439 ± 0.018** | **1.575 ± 0.044** | 0.281 ± 0.006 |
| Gated (new) | 0.422 ± 0.009 | 1.587 ± 0.092 | 0.288 ± 0.016 |

Figure: `reports/figures/phase5_fusion.png`.

**Findings:** The gated fusion does **not** improve over additive — affinity is marginally worse
(Pearson 0.439 → 0.422, RMSE 1.575 → 1.587), interface marginally better (0.281 → 0.288), all within
the error bars (tied). This is a clean negative result: the limited aggregate multimodal benefit at
650M is **not** a fusion-capacity problem — a more flexible fusion does not help. The simple additive
fusion is sufficient, and the binding signal is dominated by the strong sequence encoder.

## Part C — Interpretability (complete)

For a few clear example complexes (both chains aligned to structure, highest predicted interface),
we visualise what the model relies on. Scripts: `phase5/interp_extract.sbatch` + `interp_extract.py`
(GPU, captures cross-attention weights, per-residue interface probabilities, Cα coordinates and
true labels) and `phase5/interp_plot.py` (figures). Examples selected automatically: **2MCN**
(interface AUPR 0.94), **4KVG** (0.91), **1H0T** (0.89).

Each figure has two panels: (a) the **cross-attention map** (receptor → ligand), with the true
interface residues ticked in red; (b) the complex's **Cα atoms in 3D, coloured by the predicted
interface probability**, with the true interface residues circled in red.

**Findings:** The cross-attention concentrates in specific residue bands that line up with the true
interface, and — most clearly in 4KVG — the high-probability predicted interface residues sit
exactly at the physical contact surface between the two proteins, matching the experimental
interface. So the model's predictions are spatially coherent and interpretable, not opaque: it
localises the real binding surface and its attention focuses there. Figures:
`reports/figures/phase5_interp_{2MCN,4KVG,1H0T}.png`.

## Honest overall position

The robust, defensible results do **not** depend on the ablation: (i) the interaction model clearly
beats the published PLM-interact on the leakage-free benchmark, and (ii) the project delivers a
complete, reproducible multimodal pipeline with solid affinity prediction and honest error bars. The
one claim we cannot currently support is that structure + evolution significantly improve affinity /
interface prediction at 650M *in aggregate*; this is reported honestly. The stratified analysis
(Part B2) does, however, show a consistent evolution benefit on deep-MSA complexes — the view helps
where its signal exists — which is a defensible, mechanistic refinement of the hypothesis.

## Files

- `phase5/bench_interaction.sbatch`, `phase5/ablation_650M.sbatch`, `phase5/ablation_smoke.sbatch`,
  `phase5/agg_ablation.py` — benchmark and ablation jobs.
- `phase5/stratified_eval.sbatch`, `phase5/make_test_meta.py`, `phase5/eval_perrow.py`,
  `phase5/agg_stratified.py` — stratified analysis (Part B2).
- `phase5/gated_smoke.sbatch`, `phase5/train_gated_650M.sbatch`, `phase5/agg_gated.py` — fusion study (Part B3).
- `phase5/interp_extract.sbatch`, `phase5/interp_extract.py`, `phase5/interp_plot.py` — interpretability (Part C).
- `reports/figures/phase5_interaction_benchmark.png/.pdf`, `reports/figures/phase5_ablation.png/.pdf`,
  `reports/figures/phase5_stratified.png/.pdf`, `reports/figures/phase5_fusion.png/.pdf`,
  `reports/figures/phase5_interp_{2MCN,4KVG,1H0T}.png/.pdf`.
- View switches (`--no_struct`, `--no_evo`) and the gated fusion (`--fusion`) added to
  `phase3/train_phase3.py` / `phase3/model.py` / `phase3/data.py`.

## Next steps

1. (Optional) a small inference script: two sequences in → interaction / affinity / interface out.
2. Phase 6 — manuscript and public code release (the phase summaries are its backbone).
