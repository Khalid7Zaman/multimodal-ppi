# Phase 3 — Structural + Evolutionary modules

**Goal:** add two new "views" of each protein to the Phase 2 sequence model — a **structural**
view (3D shape, from PDBbind/RCSB structures) and an **evolutionary** view (conservation, from
MSAs) — and train the structure-aware, multi-task model on the PPB-Affinity complexes, where
structures, affinities, and interface labels all exist. Full design: `../docs/ARCHITECTURE.md`.

## Files (built step by step, like phase1/ and phase2/)

| File | What it is | Status |
|------|------------|--------|
| `struct_features.py` | **Structural module, part 1** — turn each PPB complex structure into a residue *contact graph* aligned to the sequence. Run `python phase3/struct_features.py` to self-test on real data. | ✅ built |
| `struct_module.py` | Structural module, part 2 — the graph-network layers that consume the contact map + ESM-2 features. | ⏳ next |
| `evo_features.py` | **Evolutionary module** — build MSAs from UniRef50 with MMseqs2, turn them into per-residue conservation features. | ⏳ later |
| `model.py` | The multi-task model: sequence + structure + evolution → cross-attention → interaction / affinity / interface heads. | ⏳ later |
| `train_phase3.py` + `*.sbatch` | Training loop + SLURM jobs (sanity overfit first, then full run on GPU05). | ⏳ later |

## Progress checklist
- [x] UniRef50 sequence source downloaded (`~/Projects/ppi-data/uniref50_hf`, 98 shards, ~17 GB) — feeds the evolutionary module
- [x] Structural featurizer + self-test (`struct_features.py`)
- [ ] Structural graph-network layers
- [ ] MMseqs2 database build + MSA/conservation features
- [ ] Multi-task model + training + evaluation

## How the structural data flows
```
PPB row (2 sequences + pdb id + chains)
        -> cached mmCIF structure (~/Projects/ppi-data/structures/<PDB>.cif)
        -> per-protein C-alpha contact map  (L x L, 1 = residues touch in 3D)
        -> graph-network layers fuse it with ESM-2 per-residue features
        -> cross-attention between the two proteins (as in Phase 2)
        -> heads: interaction / binding affinity / interface residues
```
