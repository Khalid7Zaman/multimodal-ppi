# Phase 2 — Core Interaction Model (ESM-2 + Cross-Attention)

**Project:** multimodal-ppi (Protein–Protein Interaction)
**Author:** Khalid Zaman, Research Assistant Professor (RAP)   **Supervisor:** Prof. Zhaoxi Sun, SUAT
**Status:** COMPLETE (35M model). 650M scale-up is the next step.

---

## Goal of Phase 2

Build our own protein–protein interaction classifier — not just run a
published one, as in Phase 0 — that takes two protein sequences and predicts
whether they interact. This is the core "interaction" head of the larger
multimodal model; binding-affinity and interface-residue heads come later.

## Model — `CrossAttnPPI`

A single, self-contained architecture (`phase2/train_ppi.py`):

1. **Encoder.** Each protein sequence is passed through **ESM-2** (a protein
   language model), **fine-tuned end-to-end** — i.e. ESM-2's own weights are
   updated during training, not frozen. This was a deliberate choice: it lets
   the language model specialise to the interaction task.
2. **Projection.** ESM-2's per-residue features are projected to a 256-dim space.
3. **Cross-attention (the key idea).** Two attention blocks let the two proteins
   "read" each other: protein A attends over protein B, and B attends over A.
   This is how the model reasons about *interaction* rather than looking at each
   protein in isolation.
4. **Symmetric pooling + head.** The two protein representations are combined in a
   way that is symmetric (A,B gives the same answer as B,A) using
   `[vA+vB, vA*vB, |vA-vB|]`, then an MLP outputs a single interaction score.

Loss: binary cross-entropy. Optimiser: AdamW. Mixed precision: bf16.

## Data

**Bernett gold-standard human PPI benchmark** (`Synthyra/bernett_gold_ppi`),
split into train / validation / test. This benchmark is deliberately built to be
*hard*: the splits are separated so a model cannot succeed by memorising which
proteins are "sticky", only by learning genuine interaction signal. Balanced
1:1 positive:negative.

- Test set: **52,048 pairs** (never seen during training).

## Training run (35M model)

| Setting | Value |
|---|---|
| Encoder | `esm2_t12_35M_UR50D` (35M params, fine-tuned) |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Max sequence length | 512 |
| Hardware | RTX 6000D (gpu05), bf16 |
| Time | ~29 min/epoch (~1.5 h total) |

**Sanity check first.** Before the real run we verified the model can *learn* by
overfitting a tiny 64-pair set: training loss → ~0 and AUROC → 1.0. This proves
the training loop, gradients, and metrics are all wired correctly.

**Validation across epochs:**

| epoch | train_loss | val AUROC | val AUPR |
|---|---|---|---|
| 1 | 0.6144 | 0.6784 | 0.6646 |
| 2 | 0.5497 | 0.6743 | 0.6653 |
| 3 | 0.5154 | 0.6742 | 0.6716 |

Training loss falls steadily while validation plateaus after epoch 1 — the
expected signature of a small (35M) encoder reaching its capacity. This is the
main motivation for scaling to the 650M encoder next.

## Result — held-out test set

**TEST AUROC 0.7033 · AUPR 0.7017** (n = 52,048).

For context, the large *published* models on this same Bernett benchmark report
**AUPR ≈ 0.69**. Our **35M** model reaches **0.70** — matching the published
baselines despite being far smaller. Test ≈ validation (in fact slightly higher),
confirming the model generalises and is not overfit.

## Files (in `phase2/`)

- `train_ppi.py` — model + data + training + evaluation (self-contained).
- `eval_ppi.py` — load a checkpoint and score a held-out CSV (reuses `train_ppi.py`).
- `sanity.sbatch` — the overfit sanity-check job.
- `train.sbatch` — the full training job (35M).
- `eval.sbatch` — the test-set evaluation job.
- `runs/ppi_35M_v1/best.pt`, `last.pt` — trained checkpoints (on cluster only; large).

Checkpoints live on the cluster only (regenerable from code); code + docs are in git.

## Next steps

1. **Scale up:** re-run the identical pipeline with the **650M** ESM-2 encoder
   (`esm2_t33_650M_UR50D`) on the RTX 6000D — expected to push past the plateau.
2. Add the **binding-affinity** head (PPB-Affinity data, already prepared in Phase 1).
3. Add the **interface-residue** head (5Å inter-chain contacts, already prepared).
4. Improve interface-label coverage via sequence alignment.
