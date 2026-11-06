# TriWeave Phase C data card: the consolidated experimental foundation

**Author:** Khalid Zaman, Research Assistant Professor (RAP).  **Supervisor:** Prof. Zhaoxi Sun, SUAT.
**Project:** TriWeave (multimodal-ppi).  2026-11-03 (block one result).

This records the result of Phase C block one, the consolidation of the experimental foundation. The
single manifest was built by `phase_c/build_experimental_manifest.py` from the Phase 1 outputs, with no
GPU, in seconds on a login node. This card reports what the foundation holds and the result of an honest
leakage check, so that every later Phase C and Phase D step draws from one documented source.

## What was built

Three files under `~/Projects/ppi-data/phase_c/` (kept out of git with the rest of the raw data):

- `experimental_manifest.csv`, one row per protein pair, with a stable pair id, an order-independent pair
  key, the source, the split, both protein ids and lengths, the interaction label, the pKd where measured,
  the structure id and chains, and flags for whether a structure, an affinity value, and interface masks
  are present.
- `proteins.csv`, one row per unique protein sequence, with the sources and splits it appears in.
- `proteins.fasta`, the unique proteins, ready for the identity-threshold leakage audit with mmseqs2.

## What the foundation holds

Interaction data, from the Bernett gold standard, trains the interaction head. It contains 274,500 pairs,
split 163,192 train, 59,260 validation, and 52,048 test, balanced at exactly 50 percent interacting.

Affinity data, from PPB-Affinity, trains the affinity head and, where interface masks exist, the interface
head. It contains 8,045 complexes, split 6,375 train, 919 validation, and 751 test. Every complex has a
measured pKd, with a median of 7.92 and a range of 1.39 to 15.70. Every complex has a cached experimental
structure, so structure coverage on this set is 100 percent. Interface masks are present on 5,269 complexes,
which is 65.5 percent overall, split 3,777 train, 818 validation, and 674 test.

Across both sources there are 20,534 unique proteins. Only one protein appears in both the interaction and
the affinity sets, so the two sources are almost entirely disjoint in their proteins.

## A deduplication note

Consolidation removed 162 duplicate complexes from the affinity set, where the same unordered pair of
sequences appeared more than once within a split. This is why the affinity counts here, 6,375, 919, and 751,
sit a little below the Phase 1 figures of 6,485, 965, and 757. The manifest keeps one row per unordered pair
per split.

## Leakage check, exact match

The check compares the test split against the train and validation splits within each source, by exact
sequence match.

The interaction data is clean. No test protein and no test pair appears in train or validation. This is the
leakage-free property the Bernett benchmark was designed to have, and it is the split the headline
interaction result rests on.

The affinity data carries a small amount of overlap. Of 1,373 test proteins, 30 (2.2 percent) also appear in
train or validation, and 1 test pair (0.1 percent) is an exact duplicate of a train or validation pair. So
the affinity split is clean at the pair level to within a single pair, but a small number of individual
proteins recur across its splits.

## What this means and the recommended follow-up

The interaction results stand on a fully leakage-free split, so they are unaffected.

The affinity split has a small protein-level overlap that was present in all of the earlier affinity numbers.
It is now measured rather than hidden, and it is small, but it means those numbers carry a slight optimistic
bias. The single exact-duplicate test pair should be dropped from the test set. The 2.2 percent protein
recurrence is best quantified fully before deciding what to do about it.

The documented follow-up is therefore an identity-threshold audit with mmseqs2 on `proteins.fasta`, which
finds near-duplicates and not only exact matches and so gives the complete picture. If that audit shows more
than a little near-duplication across the affinity splits, the clean fix is a cluster-based re-split of the
affinity data. That change would alter the affinity evaluation set, so it is one to agree with the supervisor
before it is made.

## Identity-threshold audit result (2026-11-03)

The audit was run with `phase_c/leakage_audit.sh`, which clusters the proteins with mmseqs2 at 30 percent
sequence identity and 80 percent coverage, then compares the test split against train and validation at the
cluster level. The 20,534 proteins formed 11,469 clusters, so 44.1 percent of proteins sit in a cluster with
at least one other protein.

The interaction data stays clean at the near-duplicate level. Of 3,022 test proteins only 10 (0.3 percent)
share a cluster with a train or validation protein, and no test pair matches a train or validation pair at the
cluster level. The interaction benchmark is therefore sound.

The affinity data shows substantial near-duplicate leakage that the exact-match check did not reveal. Of 1,373
test proteins, 516 (37.6 percent) share a cluster with a train or validation protein, and 62 of 751 test pairs
(8.3 percent) match a train or validation pair at the cluster level. So more than a third of the affinity test
proteins have a close homolog in training, and about one in twelve test pairs is a near-duplicate of a training
pair.

What this means. The affinity evaluation numbers reported so far, including the Phase 3 and Phase 5 figures,
were measured on this split and so carry an optimistic bias of unknown size, because the model was scored
partly on near-duplicates of what it trained on. This does not affect the interaction benchmark, which is the
headline result and is clean.

Recommended action. Produce a cluster-based re-split of the affinity data that keeps every cluster wholly
inside one split, so that no test protein has a homolog in training, and retrain the affinity model on the
clean split to get honest numbers. This will change the affinity evaluation set and will most likely lower the
reported affinity metrics. Because it changes numbers that are already on the manuscript track, it is a
decision to take to the supervisor, presented with the old and new numbers side by side so the change is
transparent.

## Sequence coverage and the synthetic-data target (2026-11-05)

Following the supervisor's instruction to analyse the sequence coverage before generating synthetic data,
`phase_c/sequence_coverage.py` measured the sequence space the experimental data covers, using the same 30
percent identity clustering.

Across both sources there are 20,534 unique proteins in 11,469 clusters. The experimental affinity set, which
the synthetic affinity data must exceed, contains 8,045 complexes built from 9,516 proteins, but those
proteins fall into only 1,814 clusters, and the complexes cover 1,428 distinct cluster-pairs. In other words
the affinity proteins are highly redundant, on the order of five proteins per cluster, which is the direct
reason the original split leaked so heavily.

From the supervisor's two requirements this gives concrete targets for the synthetic data. It should be at
least ten times the experimental size, which is at least 80,450 pairs, and it should cover more sequence space
than the experimental data, which means more than 1,814 protein clusters and more than 1,428 distinct
cluster-pairs.

Protein length ranges from 40 to 34,350 residues, with a median of 329. The few very long entries are
multi-chain complexes concatenated into one sequence, and they will be difficult for structure prediction and
physical scoring, so the structure-based part of the synthetic generation will likely need a length limit.

The seed sequence set, the 9,516 experimental affinity proteins, was written to
`~/Projects/ppi-data/phase_c/affinity_proteins.fasta`. This is the starting pool. Reaching the size and
coverage targets above is the job of block two, which pairs proteins across organisms, guided by an
interaction database.

## Clean-split retrain result (2026-11-06)

The affinity model was retrained on the leakage-free split with the frozen width-512 configuration, which is
the design the supervisor confirmed, over three seeds, with everything except the split identical to the
Phase B frozen width-512 run. Runs are in `phase3/runs/mm_esmc_clean_frozen_d512_s*`.

| Metric | Frozen width-512, original (leaky) split | Frozen width-512, clean split |
|---|---|---|
| affinity Pearson (higher better) | 0.411 +/- 0.010 | 0.421 +/- 0.052 |
| affinity RMSE (lower better) | 1.574 +/- 0.031 | 1.876 +/- 0.055 |
| interface AUPR (higher better) | 0.299 +/- 0.008 | 0.342 +/- 0.008 |

Reading, with one caveat stated first. The re-split changed which complexes sit in the test set, so this is
the honest result on a leakage-free split rather than a strict single-variable before-and-after on one fixed
test set. With that in mind, the clearest effect of removing the leakage is on the absolute affinity error:
the RMSE rose from about 1.57 to about 1.88, which is the expected direction, because the earlier figure was
flattered by near-duplicate test proteins. The ranking correlation (Pearson) is about the same, 0.411 against
0.421, but much noisier across seeds now (the spread runs from 0.373 to 0.476), which reflects a smaller and
more varied honest test set. The interface AUPR is measured on a different set of complexes after the
re-split, so it is not directly comparable and the apparent rise is not claimed as a real gain.

The honest conclusion is that the leakage mainly inflated the absolute affinity accuracy, not the ranking.
These clean numbers (Pearson 0.421, RMSE 1.876, interface AUPR 0.342) are adopted as the trustworthy affinity
baseline going forward, and the earlier RMSE is understood as optimistic. This is also the honest starting
point that the Phase D preheating stage, trained on the synthetic data, is meant to improve.

## Reproduce

From `~/Projects/multimodal-ppi`, with a pandas and numpy environment active:

    python phase_c/build_experimental_manifest.py          # block one: the manifest
    bash   phase_c/leakage_audit.sh                         # the identity-threshold audit
    python phase_c/sequence_coverage.py                     # coverage + synthetic-data targets

The outputs are written under `~/Projects/ppi-data/phase_c/`.
