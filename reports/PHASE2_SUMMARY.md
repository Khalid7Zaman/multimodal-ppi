# Phase 2 — Core Interaction Model (ESM-2 + Cross-Attention)

**Author:** Khalid Zaman, Research Assistant Professor (RAP)   **Supervisor:** Prof. Zhaoxi Sun, SUAT
**Project:** multimodal-ppi (Protein–Protein Interaction)
**Status:** COMPLETE — both the 35M and 650M models trained and evaluated on the held-out test set.

---

## Goal of Phase 2

Build our own protein–protein interaction classifier — not just run a published
one, as in Phase 0 — that takes two protein sequences and predicts whether they
interact. This is the core "interaction" head of the larger multimodal model;
binding-affinity and interface-residue heads come later.

## Model — `CrossAttnPPI`

A single, self-contained architecture (`phase2/train_ppi.py`):

1. **Encoder.** Each protein sequence is passed through **ESM-2** (a protein
   language model), **fine-tuned end-to-end** — ESM-2's own weights are updated
   during training, not frozen — so the language model specialises to the task.
2. **Projection.** ESM-2's per-residue features are projected to a 256-dim space.
3. **Cross-attention (the key idea).** Two attention blocks let the two proteins
   "read" each other: protein A attends over protein B, and B attends over A.
   This is how the model reasons about *interaction* rather than looking at each
   protein in isolation.
4. **Symmetric pooling + head.** The two protein representations are combined
   symmetrically (A,B gives the same answer as B,A) using
   `[vA+vB, vA*vB, |vA-vB|]`, then an MLP outputs a single interaction score.

Loss: binary cross-entropy. Optimiser: AdamW. Mixed precision: bf16.

## Data

**Bernett gold-standard human PPI benchmark** (`Synthyra/bernett_gold_ppi`),
split into train / validation / test. This benchmark is deliberately built to be
*hard*: the splits are separated so a model cannot succeed by memorising which
proteins are "sticky", only by learning genuine interaction signal. Balanced
1:1 positive:negative. Test set: **52,048 pairs** (never seen during training).

## Two experiments

We trained the identical pipeline at two encoder sizes, on the RTX 6000D (gpu05):

| Setting | 35M run | 650M run |
|---|---|---|
| Encoder | `esm2_t12_35M_UR50D` | `esm2_t33_650M_UR50D` |
| Epochs | 3 | 3 |
| Batch size | 16 | 8 |
| Learning rate | 2e-5 | 1e-5 |
| Max seq length | 512 | 512 |
| Time / epoch | ~29 min | ~2.6 h |

**Sanity check first (35M).** Before the real runs we verified the model can
*learn* by overfitting a tiny 64-pair set: training loss → ~0 and AUROC → 1.0.
This proves the training loop, gradients, and metrics are wired correctly.

**Validation across epochs:**

35M:

| epoch | train_loss | val AUROC | val AUPR |
|---|---|---|---|
| 1 | 0.6144 | 0.6784 | 0.6646 |
| 2 | 0.5497 | 0.6743 | 0.6653 |
| 3 | 0.5154 | 0.6742 | 0.6716 |

650M:

| epoch | train_loss | val AUROC | val AUPR |
|---|---|---|---|
| 1 | 0.5936 | 0.6867 | 0.6798 |
| 2 | 0.5273 | 0.6800 | 0.6792 |
| 3 | 0.4909 | 0.6783 | 0.6699 |

Both models' validation plateaus after epoch 1 while training loss keeps
dropping — the expected sign of the encoder reaching capacity, and (for the 650M)
of mild overfitting past epoch 1. In both cases `best.pt` is the highest-val-AUPR
epoch, which is the checkpoint we evaluate.

## Results — held-out test set (52,048 pairs)

| Model | Encoder size | Test AUROC | Test AUPR |
|---|---|---|---|
| 35M | 35M params | 0.7033 | 0.7017 |
| **650M** | 650M params | **0.7174** | **0.7093** |
| Published big models (Bernett) | large | — | ~0.69 |

**Findings:**

1. **Scaling helped.** The 650M model beats the 35M on the test set
   (AUROC +0.014, AUPR +0.008), confirming the 35M plateau was a capacity limit.
2. **Both models beat the published baselines** (~0.69 AUPR on this benchmark).
   The 650M reaches ~0.71 — above the reported state of the art.
3. **Honest generalisation.** Test ≈ validation (test is in fact slightly higher),
   with a large 52k-pair test sample, so the numbers are trustworthy and not
   overfit to the test set.

## Files (in `phase2/`)

- `train_ppi.py` — model + data + training + evaluation (self-contained).
- `eval_ppi.py` — load a checkpoint and score a held-out CSV.
- `sanity.sbatch` — the overfit sanity-check job.
- `train.sbatch` / `train_650M.sbatch` — the 35M / 650M training jobs.
- `eval.sbatch` / `eval_650M.sbatch` — the 35M / 650M test-set evaluation jobs.
- `runs/ppi_35M_v1/`, `runs/ppi_650M_v1/` — trained checkpoints (`best.pt`, `last.pt`;
  on cluster only, large).

Checkpoints live on the cluster only (regenerable from code); code + docs are in
git, mirrored to Gitee and GitHub.

## Next steps

1. Add the **binding-affinity** head (PPB-Affinity data, prepared in Phase 1).
2. Add the **interface-residue** head (5Å inter-chain contacts, prepared in Phase 1).
3. Improve interface-label coverage via sequence alignment.
4. Consider early stopping / regularisation for the 650M, which overfits past epoch 1.
