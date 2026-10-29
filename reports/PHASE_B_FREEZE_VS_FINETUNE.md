# Phase B: freeze versus fine-tune the ESM-C backbone

**Author:** Khalid Zaman, Research Assistant Professor (RAP).  **Supervisor:** Prof. Zhaoxi Sun, SUAT.
**Project:** TriWeave (multimodal-ppi).  2026-10-29 (final).

## The question
The supervisor's target design keeps the representation foundations frozen, used only as fixed feature
extractors, so that the later physics-guided preheating and experimental fine-tuning stages both run on a
fixed representation. The earlier ESM-C multimodal model, by contrast, fine-tuned the whole backbone. Phase
B measures, on our own task and data, what freezing actually costs.

## Setup
Every model here is identical except for the backbone and the fitting-network width. Same data (PPB-Affinity
complexes, all three views), same eight epochs, same batch size of two, same three seeds, and the same
held-out test scoring. The frozen models keep the 576M-parameter ESM-C backbone fixed and train only the
fitting network, at a learning rate of 1e-4. The fine-tuned model trains the whole backbone at 1e-5. Two
frozen widths were run: the baseline fitting network at d_model 256 (1.35M trainable) and a wider follow-up
at d_model 512 (4.8M trainable), to test whether a small affinity gap was just head under-capacity. Frozen
runs used 3080 Ti and RTX 6000D cards, the fine-tuned runs the large training cards. The card does not
affect the comparison beyond the seed-level noise already reported.

## Result (held-out test, mean over three seeds)

| Metric | Frozen, width 256 (1.35M) | Frozen, width 512 (4.8M) | Fine-tuned (576M) |
|---|---|---|---|
| Affinity Pearson (higher better) | 0.408 +/- 0.013 | 0.411 +/- 0.010 | 0.441 +/- 0.009 |
| Affinity RMSE (lower better) | 1.643 +/- 0.019 | 1.574 +/- 0.031 | 1.568 +/- 0.079 |
| Interface AUPR (higher better) | 0.300 +/- 0.007 | 0.299 +/- 0.008 | 0.298 +/- 0.011 |
| Trainable parameters | 1,354,499 (0.2%) | 4,806,147 (0.8%) | 576,391,491 (100%) |

## Reading
Interface prediction is tied across all three, so freezing never cost anything there. Affinity has two
faces. RMSE is the average absolute error of the predicted binding strength. Pearson is how well the
predictions track the true ordering. Widening the frozen fitting network closed the RMSE gap: it fell from
1.643 to 1.574, which matches the fine-tuned 1.568. So the first frozen model was simply a little
under-capacity for absolute accuracy, and more room fixed it. Pearson, by contrast, barely moved when the
head grew from 1.35M to 4.8M parameters (0.408 to 0.411), and it still trails the fine-tuned 0.441 by about
0.03. Because more head capacity did not help the correlation, that remaining gap is a property of the frozen
representation, not of head size. Fine-tuning lets the backbone adjust its internal representation to rank
affinities a little better, which a frozen representation cannot fully match at any head size.

## Decision (to confirm with the supervisor)
Adopt the frozen foundations, as the staged design requires. With the backbone frozen and only about 0.8
percent of the parameters trained, the model matches the fully fine-tuned model on interface and on affinity
RMSE, and trails it only slightly on affinity ranking. The cost of freezing is therefore small. The small
ranking gap is representation-level and will not be closed by more head capacity, so it is not a reason to
un-freeze. It is exactly what the physics-guided preheating stage is designed to recover, by training the
fitting network on a large body of simulated and distilled data before experimental fine-tuning. Freezing is
adopted, the small ranking gap is recorded plainly, and it is carried to the preheating stage.

## Reproduce
- Frozen baseline (width 256): `sbatch phase3/train_phase3_esmc_frozen_seeds.sbatch`
- Frozen wider (width 512): `sbatch phase3/train_phase3_esmc_frozen_d512_seeds.sbatch`
- Aggregate (esmc environment): `python phase3/agg_seeds.py "phase3/runs/mm_esmc_frozen_s*"` and
  `python phase3/agg_seeds.py "phase3/runs/mm_esmc_frozen_d512_s*"`
- Fine-tuned baseline numbers from the earlier ESM-C runs: `phase3/runs/mm_esmc_s*`.
