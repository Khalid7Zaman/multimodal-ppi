# TriWeave Phase C block two: the candidate interacting-pair set

**Author:** Khalid Zaman, Research Assistant Professor (RAP).  **Supervisor:** Prof. Zhaoxi Sun, SUAT.
**Project:** TriWeave (multimodal-ppi).  2026-11-09.

Block two builds the candidate protein pairs that the synthetic data will be generated on, following the
supervisor's direction that the work be general across organisms and that the synthetic set be at least ten
times the experimental data with broader sequence coverage. The whole pipeline is CPU only and leaves the raw
data untouched.

## Source

STRING version 12.0, the reported-interaction database, is used to choose plausible interacting pairs rather
than random ones. The physical subnetwork is taken, meaning proteins that physically bind rather than ones
that are merely associated, and only high-confidence edges are kept, with a STRING combined score of at least
700 out of 1000. Fifteen organisms span the tree of life: human, mouse, zebrafish, fruit fly and nematode
among animals, Arabidopsis as a plant, budding yeast and fission yeast as fungi, Escherichia coli, Bacillus
subtilis, Mycobacterium tuberculosis, Pseudomonas aeruginosa and Synechocystis among bacteria, Methanocaldococcus
jannaschii as an archaeon, and Dictyostelium as an amoeba. For each organism the physical links and the protein
sequences were downloaded.

## Pipeline

Three steps under `phase_c/`, each writing into `~/Projects/ppi-data/`.

1. `block2_fetch_string.sh` downloads the STRING physical links and sequences for each organism.
2. `block2_build_candidates.py` keeps the high-confidence physical edges as positive, plausible-binder pairs,
   attaches the sequences, pools across organisms, and removes the duplicate directions that STRING lists for
   every edge. Result: 343,246 unique candidate pairs from 54,544 proteins.
3. `block2_deleak.sh` clusters the candidate proteins together with the evaluation test proteins, which are the
   clean affinity test split plus the Bernett interaction test split, 3,880 proteins in all, at 30 percent
   identity, and drops any candidate pair whose protein shares a cluster with a test protein. This keeps the
   synthetic data from leaking into the evaluation.

## Result

After de-leaking, 224,139 candidate interacting pairs remain, built from 44,117 proteins in 20,335 clusters,
covering 123,026 distinct cluster-pairs across all fifteen organisms. De-leaking removed 119,107 pairs, about
35 percent, that resembled a test protein, which shows the screen is working.

All three targets set by the supervisor are met with wide margins.

| Target | Required | Achieved |
|---|---|---|
| set size | at least 80,450 pairs (ten times experimental) | 224,139 |
| protein clusters | more than 1,814 (experimental) | 20,335 |
| distinct cluster-pairs | more than 1,428 (experimental) | 123,026 |

So the candidate set is about 28 times the size of the experimental affinity data, with roughly 11 times the
protein-cluster coverage and far broader interaction-space coverage, and it carries none of the evaluation test
set.

## Output and what comes next

The candidate set is `~/Projects/ppi-data/phase_c/candidate_pairs_clean.csv`, with one row per pair giving the
organism, the STRING identifiers, the stable sequence identifiers, the confidence score, and the two lengths.
The sequences are in `candidate_proteins.fasta`.

These are the plausible interacting pairs that block three will label with synthetic values, from the physical
simulation, which is the MM/GBSA work with the colleague and the EnsPlex docking data the supervisor is
bringing, and from distillation of a strong released predictor, which is the teacher model the supervisor is
selecting. The full set is the pool. The subset sent to the expensive structure-based simulation can be sampled
from it to match whatever compute budget the simulation side sets.

## Reproduce

From `~/Projects/multimodal-ppi`, with the `esmc` environment active:

    bash   phase_c/block2_fetch_string.sh
    python phase_c/block2_build_candidates.py
    bash   phase_c/block2_deleak.sh

Outputs are written under `~/Projects/ppi-data/phase_c/` and `~/Projects/ppi-data/string/`, which are kept out
of git with the rest of the raw data.
