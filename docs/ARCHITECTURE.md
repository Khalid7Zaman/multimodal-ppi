# Architecture — multimodal-ppi

Design reference for the model. Expands Section 3 of `PROJECT_STATE.md` and Section 6 of
`docs/PROPOSAL.md`, and records how each stage is actually implemented as the phases land.
Last updated: 2026-09-17 (Phase 3 kickoff).

The model takes **two protein sequences** and predicts **(1)** whether they interact,
**(2)** their binding affinity, and **(3)** their interface residues — with residue-level
interpretability. Known structures are an *input*, never predicted (bound-structure
prediction is the group's separate work).

## Five stages

### Stage 1 — Input layer
Two protein sequences (A, B). A single **shared** encoder reads both, so A,B and B,A are
treated symmetrically. Structure/MSA are optional extra inputs used where available.

### Stage 2 — Multimodal encoder (three views per protein)
- **Sequence module** — ESM-2 (fine-tuned end-to-end; ESM-3 preferred later), per-residue
  embeddings → projected to `d_model`. *Implemented in Phase 2.*
- **Structural module** — an experimental complex (PDBbind / RCSB mmCIF) → per-protein
  **residue contact graph** (nodes = residues, edges = spatial contacts) → graph network;
  fused with the sequence embeddings. *Phase 3.*
- **Evolutionary module** — an MSA per protein → per-residue **conservation** features
  (and, where used, co-variation) → conservation encoder; fused in. *Phase 3.*

### Stage 3 — Cross-protein reasoning engine
Two-way **cross-attention**: every residue of A attends over B and vice versa, producing a
residue-level **interaction map** (the interpretability output). *Implemented in Phase 2 as
`a2b` / `b2a` MultiheadAttention.*

### Stage 4 — Multi-task head
Shared representation → three heads:
- **Interaction** (binary) — *Phase 2, done.*
- **Binding affinity** (pKd regression) — *Phase 4 (brought forward into Phase 3 work).*
- **Interface residues** (per-residue binary) — *Phase 4 (brought forward).*

### Stage 5 — Training on experimental data
Supervised throughout. Interaction labels from public PPI datasets; affinities + structures
from PDBbind (accessed via PPB-Affinity + RCSB). Mixed precision (bf16) + gradient
checkpointing to fit the big encoder on the GPUs.

## Phase 2 baseline — `CrossAttnPPI` (implemented, `phase2/train_ppi.py`)
```
seq A ─┐                                     ┌─ vA = pool(norm(hA + A→B attn))
       ├ ESM-2 (shared, fine-tuned) → proj ──┤                                  → [vA+vB, vA*vB, |vA−vB|] → MLP → interaction logit
seq B ─┘                                     └─ vB = pool(norm(hB + B→A attn))
```
Symmetric pooling makes the score order-invariant. Loss BCEWithLogits, AdamW, bf16.
Result (Bernett held-out, 52,048 pairs): 650M **AUROC 0.7174 / AUPR 0.7093** (> ~0.69 baseline).

## Phase 3 plan — add structural + evolutionary modules (on PPB complexes)
Structures and feasible MSAs exist for the ~6.5k **PPB-Affinity** complexes, not the 163k
sequence-only Bernett pairs — so the two new modules are developed and validated on PPB,
and the affinity + interface heads (Phase 4) are brought forward to train the structure-aware
model end-to-end where all labels co-exist. The Phase 2 sequence model remains the interaction
head.

Data already prepared (Phase 1, see `phase1/DATA_README.md`):
- `ppb_{train,val,test}_iface.csv` — receptor/ligand sequences, `pKd`, `pdb`,
  `receptor_chains`, `ligand_chains`, and `iface_query`/`iface_text` (0/1 per residue, 5 Å contacts).
- Structures cached on the cluster: `~/Projects/ppi-data/structures/*.cif`.

Design decisions to finalise at build time (recorded here as they are made):
- Structural graph: contact cutoff (Cα–Cα ~8 Å or any-atom 5 Å), node features (ESM per-residue
  emb + optional DSSP), edge features (distance bins), GNN type (GCN/GAT/GVP).
- Evolutionary features: MSA source/tool on the cluster (hhblits/jackhmmer + DB, or MSA-Transformer),
  conservation representation (per-residue entropy / PSSM).
- Fusion point: add structural + evolutionary features to the projected sequence embeddings
  *before* cross-attention, so the interaction map benefits from all three views.
- Losses: BCE (interface, masked to labelled residues) + MSE/Huber (pKd) [+ BCE interaction],
  weighted; report affinity RMSE/Pearson and interface AUPR/F1.

## Related docs
- `PROJECT_STATE.md` — single source of truth (status, decisions log).
- `docs/PROPOSAL.md` — full research proposal (v10).
- `reports/PHASE0_SUMMARY.md`, `PHASE1_SUMMARY.md`, `PHASE2_SUMMARY.md` — phase write-ups.
- `phase1/DATA_README.md` — dataset details.
- `WORKFLOW.md` — git sync cheat-sheet (pull at start, push at end).
