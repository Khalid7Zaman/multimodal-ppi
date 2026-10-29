# Phase B: freeze versus fine-tune the ESM-C backbone

**Author:** Khalid Zaman, Research Assistant Professor (RAP).  **Supervisor:** Prof. Zhaoxi Sun, SUAT.
**Project:** TriWeave (multimodal-ppi).  2026-10-29.

## The question
The supervisor's target design keeps the representation foundations frozen, used only as fixed feature
extractors, so that the later physics-guided preheating and experimental fine-tuning stages both run on a
fixed representation. The earlier ESM-C multimodal model, by contrast, fine-tuned the whole backbone. Phase
B measures, on our own task and data, what freezing actually costs.

## Setup
Both models are identical in every way except the backbone. Same data (PPB-Affinity complexes, all three
views), same eight epochs, same batch size of two, same three seeds, and the same held-out test scoring.
The frozen model keeps the 576M-parameter ESM-C backbone fixed and trains only the fitting network (the
projection, structure and evolution encoders, the cross-attention and the three heads). The frozen runs use
a learning rate of 1e-4, ten times the fine-tuned 1e-5, because only the randomly initialised fitting
network trains and the low rate exists to protect pretrained weights during fine-tuning. The frozen runs
ran on a 3080 Ti (12 GB), the fine-tuned runs on the larger training cards. The card does not affect the
comparison beyond the seed-level noise already reported.

## Result (held-out test, mean over three seeds)

| Metric | Frozen (fitting network only) | Fine-tuned (full backbone) |
|---|---|---|
| Affinity Pearson (higher better) | 0.408 +/- 0.013 | 0.441 +/- 0.009 |
| Affinity RMSE (lower better) | 1.643 +/- 0.019 | 1.568 +/- 0.079 |
| Interface AUPR (higher better) | 0.300 +/- 0.007 | 0.298 +/- 0.011 |
| Trainable parameters | 1,354,499 (0.2%) | 576,391,491 (100%) |

Frozen per-seed affinity Pearson: 0.420, 0.394, 0.409.

## Reading
Interface prediction is tied: frozen 0.300 against fine-tuned 0.298, well inside the error bars. On binding
affinity the fine-tuned model is modestly better, by about 0.033 in Pearson and 0.075 in RMSE, a gap a
little larger than the error bars but small in absolute terms. The frozen model reached this while training
only 0.2 percent of the parameters and running on a small gaming GPU. Most of the useful signal is therefore
already present in the frozen ESM-C features, and fine-tuning the full backbone buys only a small affinity
gain and nothing on interface.

## Caveat and follow-up
The frozen fitting network is tiny, only 1.35M parameters, so the small affinity gap may be under-capacity
rather than a limit of frozen features. A follow-up run doubles the fitting-network width to d_model 512
(about four times the trainable parameters, still around one percent of the backbone) with everything else
unchanged (`phase3/train_phase3_esmc_frozen_d512_seeds.sbatch`). If the affinity gap shrinks, freezing is
confirmed with little cost. If it does not, the gap is intrinsic to the frozen representation and is a
target for the physics-guided preheating stage instead.

## Decision (to confirm with the supervisor)
Adopt the frozen foundations, as the staged design requires. The cost is small (no interface loss, a modest
affinity loss) and the benefits are large (modularity, a fixed representation for the preheating and
fine-tuning stages, and training that fits on small, available GPUs). The affinity gap is recorded plainly
and is treated as a target for more fitting-network capacity and for the physics-guided preheating stage,
not as a reason to fine-tune the backbone.

## Reproduce
- Frozen baseline: `sbatch phase3/train_phase3_esmc_frozen_seeds.sbatch`
- Frozen wider follow-up: `sbatch phase3/train_phase3_esmc_frozen_d512_seeds.sbatch`
- Aggregate: `python phase3/agg_seeds.py "phase3/runs/mm_esmc_frozen_s*"` (esmc environment)
- Fine-tuned baseline numbers from the earlier ESM-C runs: `phase3/runs/mm_esmc_s*`.
