# Phase 3 — Structural and Evolutionary Modules (Multimodal, Multi-Task Model)

**Author:** Khalid Zaman, Research Assistant Professor (RAP)   **Supervisor:** Prof. Zhaoxi Sun, SUAT
**Project:** multimodal-ppi (Protein–Protein Interaction)   ·   2026-09-29
**Status:** COMPLETE — structural and evolutionary modules built; multiple sequence alignments generated for all 9,516 proteins; the multimodal multi-task model trained and evaluated. A first controlled ablation shows the evolutionary module improves every task.

---

## Goal of Phase 3

Extend the Phase 2 sequence model into a genuine multimodal model by adding two further
views of each protein — its 3D **structure** and its **evolutionary** history — and train a
multi-task model on the PPB-Affinity complexes that predicts **binding affinity** and
**interface residues**. This is where the project's central hypothesis is first tested: that
joining sequence, structure, and evolution predicts more accurately than sequence alone.

We build the structural and evolutionary modules on the PPB-Affinity complexes (where
structures, affinities, and interface labels all co-exist) and bring the Phase 4 affinity and
interface heads forward into this phase. The interaction head remains the Phase 2 sequence
model, since the 274,500 Bernett interaction pairs are sequence-only and MSAs are infeasible at
that scale.

## What we built

### 1. Structural module

- `struct_features.py` — from each complex's experimentally determined structure (cached RCSB
  mmCIF), we build a per-protein **C-alpha residue contact graph** (contacts within 8 Å),
  aligned one-to-one to the sequence. About 60% of complexes align cleanly; the rest fall back
  to a sequence-only view.
- `struct_module.py` — a **plain-PyTorch graph convolutional network** that consumes the contact
  map. We built this deliberately **without** torch-geometric to avoid fragile compiled-extension
  installs on the CUDA 12.8 / Blackwell stack.

### 2. Evolutionary module

- For each of the **9,516** unique PPB proteins we search **UniRef50** (58,869,447 sequences)
  with MMseqs2, assemble a multiple sequence alignment (MSA), and read off per-residue
  conservation.
- `evo_features.py` — MSA (`.a3m`) → **22-dimensional** per-residue features (20 amino-acid
  frequencies + conservation + gap fraction), plus a conservation encoder.
- **Outcome:** MSAs for **all 9,516 proteins**; **844,884** unique homologs found; **median 945**
  sequences per alignment (mean 638); 100% are real alignments (≥ 2 sequences). This is deep,
  publication-grade evolutionary coverage. Depth and data-loader integrity are recorded in
  `reports/PHASE3_MSA_VERIFY.txt`.

*Reproducible pipeline (engineering note).* We search the full UniRef50 once, collect the hit
sequences into a reduced database, rebuild it with compact indexing, then assemble the MSAs. The
MSA-assembly step has a large fixed memory footprint (~240 GB) and is run single-threaded on the
RTX 6000D node (gpu05); the search and database steps run multi-threaded on ordinary nodes. The
pipeline is resumable at each stage.

### 3. Multimodal, multi-task model

- `model.py` — **MultiModalPPI**: sequence (ESM-2) + structure (contact-graph network) +
  evolution (conservation encoder) are joined and passed through a **two-way cross-attention**
  layer, whose shared output feeds a **binding-affinity** head (pKd, MSE loss) and a
  **interface-residue** head (per-residue masked BCE loss).
- `data.py` — assembles, per complex and for both proteins, token-aligned sequence + structure +
  evolution features and the affinity/interface targets.

## Data

PPB-Affinity complexes prepared in Phase 1: **6,485 / 965 / 757** train / validation / test
complexes with measured pKd; interface residues from 5 Å inter-chain contacts; **9,516** unique
proteins across all complexes.

## Experiment — does evolution help?

We ran a **clean ablation**: the identical model (35M ESM-2 encoder, 5 epochs, batch 2,
lr 1e-5) trained twice, differing **only** in whether the evolutionary features were active.
Both runs on the RTX 6000D (gpu05).

## Results — held-out validation set (965 complexes)

| Model | Affinity RMSE ↓ | Affinity Pearson ↑ | Interface AUPR ↑ |
|---|---|---|---|
| Sequence + Structure | 2.032 | 0.293 | 0.256 |
| **Sequence + Structure + Evolution** | **1.820** | **0.343** | **0.271** |
| Change from adding evolution | −0.21 (better) | +0.050 (better) | +0.015 (better) |

**Findings:**

1. **Evolution helps on every task.** Adding the evolutionary module lowers the affinity error
   (RMSE 2.032 → 1.820), raises the affinity correlation (Pearson 0.293 → 0.343, +17% relative),
   and improves interface detection (AUPR 0.256 → 0.271). The improvement is consistent across all
   three measures.
2. **This supports the central hypothesis** — that joining sequence, structure, and evolution
   beats using less. It is the first direct evidence that the multimodal design works.
3. **Honest scope.** These are single runs of the small 35M encoder at 5 epochs, on the validation
   split. The *direction* is trustworthy (every metric moves the right way); the *magnitude* and
   statistical strength come from the larger, repeated runs planned next. We do not overstate a
   modest first result.

## Files (in `phase3/`)

- `struct_features.py`, `struct_module.py` — structural featurizer and graph network.
- `collect_ppb_seqs.py`, `build_uniref_db.sh/.sbatch`, `msa_search.sh/.sbatch`,
  `msa_finish2.sbatch`, `msa_pack.sbatch`, `verify_msa.sbatch` — the MSA pipeline (search,
  reduce, assemble, verify).
- `evo_features.py` — conservation features and encoder.
- `model.py`, `data.py` — multimodal model and multi-task dataset.
- `train_phase3.py`, `sanity_phase3.sbatch`, `train_phase3.sbatch` — training and sanity check.
- `progress.sh` — progress monitor.
- `reports/PHASE3_MSA_VERIFY.txt` — MSA depth distribution and data-loader verification.

MSAs and databases live on the cluster under `~/Projects/ppi-data/` (kept out of git). Code and
documentation are in git, mirrored to Gitee and GitHub.

## Next steps

1. **Definitive training run** — the larger **650M** encoder, more epochs, and repeated random
   seeds so the results carry error bars.
2. **Full ablation figure** — sequence-only → + structure → + evolution, on one fixed split.
3. **Benchmarking (Phase 5)** — against PLM-interact on interaction, and against structure-based
   scoring on affinity; plus the interpretability figures (cross-attention interaction maps and
   predicted interface residues shown on the 3D structures).
