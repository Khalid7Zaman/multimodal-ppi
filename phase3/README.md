# Phase 3 — Structural + Evolutionary modules (multimodal, multi-task model)

**Goal:** add a **structural** view (3D contact graphs from PDBbind/RCSB) and an **evolutionary**
view (MSA conservation) to the Phase 2 sequence model, and train a multi-task model on the PPB
complexes that predicts **binding affinity** and **interface residues**. Design: `../docs/ARCHITECTURE.md`.

## Files

| File | What it is | Status |
|------|------------|--------|
| `struct_features.py` | Structural featurizer — mmCIF → per-protein C-alpha contact graph, aligned to sequence. | ✅ tested |
| `struct_module.py` | `StructEncoder` — plain-PyTorch GCN that consumes the contact map. | ✅ tested |
| `build_uniref_db.sh` / `.sbatch` | Build the UniRef50 MMseqs2 search database from the CSV shards. | ✅ done |
| `collect_ppb_seqs.py` | De-duplicate PPB proteins → `ppb_queries.fasta` (9,516 proteins). | ✅ done |
| `msa_search.sh` / `.sbatch` | Search PPB proteins vs UniRef50 → one `.a3m` per protein. **Needs a big-memory node (gpu05/cpu05, ≥320 GB)** — the alignment step needs ~258 GB (see the .sbatch header). | ✅ proven (test); full run in progress |
| `evo_features.py` | MSA `.a3m` → per-residue conservation (20 freqs + conservation + gap) + `ConservationEncoder`. | ✅ tested |
| `model.py` | `MultiModalPPI` — sequence + structure + evolution → cross-attention → interaction / affinity / interface heads. | ✅ tested |
| `data.py` | `PPBComplexData` — assembles sequence + structure + MSA-conservation + targets per complex, token-aligned. | ✅ built (validate after full MSAs) |
| `train_phase3.py` | Training loop for the affinity + interface heads; reports RMSE / Pearson / AUPR. | ✅ built |
| `sanity_phase3.sbatch` / `train_phase3.sbatch` | SLURM GPU jobs: overfit sanity check, then full training on gpu05. | ✅ built |

## Progress checklist
- [x] Structural featurizer + graph module (tested)
- [x] UniRef50 database built; MMseqs2 installed
- [x] PPB query set (9,516 proteins)
- [x] MSA search **proven** on a 20-protein test (≈332 homologs/protein); **full run in progress**
- [x] Conservation features + encoder (tested)
- [x] Multi-task model (tested end-to-end)
- [x] Data loader + training script (built)
- [ ] Full MSAs finished → validate `data.py` → GPU **sanity overfit** → **full training** on gpu05
- [ ] `reports/PHASE3_SUMMARY.md` + update `PROJECT_STATE.md`

## The pipeline, end to end
```
sequences ─ ESM-2 ─┐
structures ─ contact graph ─ StructEncoder ─┤→ per-residue features → cross-attention → heads:
MSAs ─ conservation ─ ConservationEncoder ──┘        affinity (pKd) · interface (per residue)
```

## MSA memory lesson (2026-09-21)
The mmseqs **alignment** step needs ~258 GB RAM for the full 58.9M UniRef50 and is **not** reduced
by `--split`, `--split-memory-limit`, or `--db-load-mode` (those only bound the prefilter). Run it
on a big-memory node (`gpu05`, 515 GB) with `--mem 320G`. Verified working.
