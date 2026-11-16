# TriWeave Phase C: PPI-pair importance score for the synthetic candidate set

**Author:** Khalid Zaman, Research Assistant Professor (RAP).  **Supervisor:** Prof. Zhaoxi Sun, SUAT.
**Project:** TriWeave (multimodal-ppi).  2026-11-16.

This records the pair-importance analysis the supervisor asked for, with numerical evidence. It is the
second of the three analyses that gate large-scale synthetic data generation. Its purpose is to rank the
candidate interacting pairs by how valuable each one is to generate a synthetic label for, so the
generation budget is spent on the pairs that matter most rather than on a random sample.

## Method

The input is the de-leaked candidate set from block two, 224,139 interacting pairs over 44,117 proteins
across 15 organisms. Each pair receives a score built from three transparent axes. Each axis is turned
into a percentile across all pairs, so no axis dominates by its raw scale, and the three are averaged
with equal weight by default. The axes are kept separate in the output so the weighting can be changed
and defended.

Reliability is the STRING combined physical score of the pair, from 700 to 1000. A higher score means
the interaction is better supported, so a synthetic label on it is more trustworthy.

Centrality is how central the two proteins are in the candidate interaction network. The pairs define a
graph, and for each protein we measure its degree and its score-weighted PageRank, the standard measures
of how much of a hub a node is. Hub proteins are enriched for essential and disease-related functions, so
modelling their interactions well has more downstream value.

Novelty is how far the two proteins sit from the experimental proteins in the model's own ESM-C
representation, measured in the same PCA frame used for the coverage analysis, in units of the typical
spacing between experimental proteins. A value near 1 means the protein sits about as far from the
experimental set as experimental proteins sit from each other, and larger values mean it reaches into new
regions. Novel pairs are the ones that extend the sequence space beyond the experimental envelope.

Script: `phase_c/pair_importance.py`. It reads only the candidate pairs and the embeddings already
computed for the coverage analysis, and writes `candidate_pairs_scored.csv` and `pair_importance.png`.
It runs on a login node, CPU only, and changes none of the existing data or results.

## Result

The ranking works. Selecting the top pairs by importance gives a set that is both more novel and more
central than a random selection of the same size. The top 5,000 pairs have a mean novelty of 1.88 against
1.19 for a random 5,000, which is about 1.6 times more novel, and a mean PageRank of 1.41e-04 against
7.82e-05, which is about 1.8 times more central. The advantage is largest at the top of the list and
narrows as the selection grows, which is the expected behaviour, since the best pairs are used up first.
Even at 80,000 pairs the top selection stays clearly ahead of random on both novelty and centrality.

The very top of the list is strong on every axis at once, not by trading one off against another. The
overall top 1,000 pairs sit at the 0.89 percentile on reliability, the 0.92 percentile on centrality, and
the 0.92 percentile on novelty. So the highest-ranked pairs are the ones that are well supported,
central, and new all together.

The underlying spread is healthy. Protein degree in the candidate network runs from a median of 4 up to a
maximum of 295, so there is a real range of hubs to prefer. Protein novelty has a median of 1.23 and a
95th percentile of 2.34, so a meaningful tail of candidates sits well beyond the experimental set and can
be targeted deliberately.

## What this means

This gives the project a defensible order in which to generate synthetic labels. Instead of sampling the
224,139 candidate pairs at random, the generation can start from the top of the ranked list and move
down, spending the simulation and distillation budget on the pairs that are the most reliable, the most
central, and the most novel. It also connects directly to the coverage analysis. The coverage work found
that the candidate set covers the experimental space well but only partially extends beyond it. The
novelty axis here is the lever for that extension, because sorting by importance, or by novelty alone,
pulls the selection toward the pairs that stretch the space rather than merely fill it. The weights on the
three axes are exposed as settings, so the balance between reliability, centrality, and novelty can be
tuned to match the supervisor's priority once the generation budget is fixed.

## Numbers at a glance

- Candidate pairs scored: 224,139 over 44,117 proteins and 15 organisms.
- Network: protein degree median 4, maximum 295. PageRank median 2.27e-05, maximum 8.78e-04.
- Novelty (protein to nearest experimental, in experimental spacings): median 1.23, 95th percentile 2.34.
- Top versus random selection, mean novelty: 5,000 pairs 1.88 against 1.19; 10,000 1.80 against 1.20;
  20,000 1.72 against 1.20; 40,000 1.60 against 1.20; 80,000 1.47 against 1.19.
- Top versus random selection, mean PageRank: 5,000 pairs 1.41e-04 against 7.82e-05, falling to
  9.83e-05 against 7.78e-05 at 80,000.
- Top 1,000 pairs, mean percentile per axis: reliability 0.89, centrality 0.92, novelty 0.92.

## Reproduce

From `~/Projects/multimodal-ppi` in the esmc environment:

    python phase_c/pair_importance.py \
      --pairs ~/Projects/ppi-data/phase_c/candidate_pairs_clean.csv \
      --exp   ~/Projects/ppi-data/phase_c/emb_experimental \
      --syn   ~/Projects/ppi-data/phase_c/emb_synthetic \
      --out   ~/Projects/ppi-data/phase_c

The weights can be changed with `--w_reliability`, `--w_centrality`, and `--w_novelty`.
