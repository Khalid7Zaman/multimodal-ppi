# Phase 3 — Structural + Evolutionary modules

**Goal:** add two new "views" of each protein to the Phase 2 sequence model — a **structural**
view (3D shape, from PDBbind/RCSB structures) and an **evolutionary** view (conservation, from
MSAs) — and train the structure-aware, multi-task model on the PPB-Affinity complexes, where
structures, affinities, and interface labels all exist. Full design: `../docs/ARCHITECTURE.md`.

## Files (built step by step, like phase1/ and phase2/)

| File | What it is | Status |
|------|------------|--------|
| `struct_features.py` | **Structural module, part 1** — turn each PPB complex structure into a residue *contact graph* aligned to the sequence. Run `python phase3/struct_features.py` to self-test on real data. | ✅ built |
| `struct_module.py` | Structural module, part 2 — the GCN graph-network layers (`StructEncoder`) that consume the contact map + ESM-2 features. Run `python phase3/struct_module.py` to self-test. | ✅ built |
| `build_uniref_db.sh` + `.sbatch` | **Evolutionary module, step 1** — convert the 98 UniRef50 CSV shards → FASTA and `mmseqs createdb` to build the search database (SLURM CPU job). | ✅ built |
| `collect_ppb_seqs.py` | **Evolutionary module, step 2** — de-duplicate PPB receptor/ligand sequences → `ppb_queries.fasta` (the proteins we need MSAs for). | ✅ built |
| `msa_search.sh` + `.sbatch` | **Evolutionary module, step 3** — `mmseqs search` the 9,516 PPB proteins vs UniRef50 → one MSA (`.a3m`) per protein (SLURM CPU job). | ✅ built |
| `evo_features.py` | **Evolutionary module, step 4** — read each MSA → per-residue conservation features (20 aa freqs + conservation + gap) + `ConservationEncoder`. Run `python phase3/evo_features.py` to self-test. | ✅ built |
| `model.py` | The multi-task model: sequence + structure + evolution → cross-attention → interaction / affinity / interface heads. | ⏳ later |
| `train_phase3.py` + `*.sbatch` | Training loop + SLURM jobs (sanity overfit first, then full run on GPU05). | ⏳ later |

## Progress checklist
- [x] UniRef50 sequence source downloaded (`~/Projects/ppi-data/uniref50_hf`, 98 shards, ~17 GB) — feeds the evolutionary module
- [x] Structural featurizer + self-test (`struct_features.py`)
- [x] Structural graph-network layers (`struct_module.py`, `StructEncoder`)
- [x] UniRef50 database build job (`build_uniref_db.sh` / `.sbatch`)
- [x] PPB query set (`collect_ppb_seqs.py` → 9,516 unique proteins)
- [x] MSA search job (`msa_search.sh` / `.sbatch`)
- [x] Per-residue conservation features + encoder (`evo_features.py`)
- [ ] Run the DB build + MSA search on the cluster (jobs ready)
- [ ] Multi-task model (`model.py`) + training
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
