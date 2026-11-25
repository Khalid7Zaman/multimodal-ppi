# TriWeave Phase C: affinity-teacher benchmark, PPLM-Affinity and Pythia-PPI

**Author:** Khalid Zaman, Research Assistant Professor (RAP).  **Supervisor:** Prof. Zhaoxi Sun, SUAT.
**Project:** TriWeave (multimodal-ppi).  2026-11-25.

This is the second distillation-teacher benchmark, run on the two models named in the group discussion.
The first benchmark tested a released sequence-only affinity model and rejected it, so the question here
was whether either of these two is accurate enough and cheap enough to label the candidate pairs. The
protocol is the one used in the first benchmark, on the same data, so the three teachers are directly
comparable. PPLM-Affinity passes. Pythia-PPI is accurate but answers a different question and cannot be
applied to the candidate set at all.

## Method

Both models were installed inside the project folder on the cluster and kept out of git, since their
weights are large and one of the two is non-commercially licensed. All runs went through Slurm to compute
nodes.

**PPLM-Affinity** is a paired-sequence language model, a 650-million-parameter backbone with an affinity
head, released with five cross-validation models that the authors' script averages. It takes two sequences
and returns one binding free energy, so it can be applied to the candidate pairs. It was fine-tuned on
PPB-Affinity, which is also the source of our experimental affinity set; of our 8,207 measured complexes,
8,204 appear somewhere in its training folds. Averaging all five models over such a complex would measure
memorisation, so accuracy here is reported in the leakage-aware way: each complex is scored only by the
fold model that did not train on it, using the fold membership published by the authors. The optimistic
all-model figure is reported alongside it purely to show the size of the gap. Run through
`phase_c/teacher_pplm_affinity.py`.

**Pythia-PPI** is structure-based. It reads a PDB file of a complex, builds a neighbour graph, and predicts
the change in binding affinity caused by each single-point mutation. It was validated against SKEMPI 2.0,
computing the measured change from the wild-type and mutant dissociation constants as ddG = RT ln(Kd_mut /
Kd_wt), at the temperature recorded with each measurement. Run through `phase_c/teacher_pythia_ppi.py`.

Three questions were asked of each model, as before: what one label costs, how widely its predictions
spread over our candidate pairs, and how well it agrees with laboratory measurements.

## Result

**PPLM-Affinity, accuracy.** Against measured pKd on 500 experimental complexes, scored only by models
that never saw them: Pearson -0.755, Spearman -0.760. The sign is correct, since a lower predicted free
energy means stronger binding and a higher pKd. The strength holds across the splits, from -0.699 on the
test portion to -0.792 on validation. Scoring the same complexes with all five models, the way the
released script does, gives -0.845, so memorisation accounts for about 0.09 of apparent accuracy. For
comparison, the teacher rejected in the first benchmark scored -0.146 on these same complexes.

**PPLM-Affinity, spread.** Over 200 random candidate pairs the predictions run from -12.11 to -7.96, a
range of 4.15 with a standard deviation of 0.62. The rejected teacher spanned 0.3 and sat on its training
mean. This model discriminates between pairs.

**PPLM-Affinity, cost.** On one graphics card a pair takes 2.71 seconds in the full setting, five fold
models in both chain orders, which is ten forward passes. Predictions from the two chain orders differ by
at most 0.000015, so the second order adds nothing and can be dropped. In the cheapest setting, one fold
model and one chain order, a pair costs about 0.27 seconds, which puts the full candidate set at roughly
17 hours on a single card, or about 3.5 days using all five models. On processors alone the same work takes
20.5 seconds per pair, about 53 days, so a graphics card is worth requesting for this.

**Pythia-PPI.** Forty SKEMPI complexes were scored in 78 seconds on one card, about two seconds per
structure, each covering every possible single-point mutation, roughly ten thousand per structure. Of the
mutations whose effect has been measured, 483 matched a scored mutation. Agreement with the laboratory
values was Pearson +0.897 and Spearman +0.892 pooled, with a per-complex Pearson median of +0.930 over the
eleven complexes carrying at least ten measured mutations. Predicted and measured ranges match closely,
-5.35 to +8.21 against -5.04 to +8.81 kcal/mol.

## What this means

PPLM-Affinity is the first affinity teacher we have tested that is usable. It is accurate enough to rank
pairs, cheap enough to label the whole candidate set within a day on one card, and it needs sequences
only. It is not, however, a replacement for simulation as a source of trustworthy values. Its accuracy
against measurement, around 0.75, is good for soft targets but not for ground truth, and it was trained on
the same affinity dataset our own experimental set comes from, so distilling from it risks propagating
that dataset's biases into our synthetic labels. The sound use is as preheating labels in staged training,
with MM/GBSA providing the anchor values that the model is finally trained and judged against. The direct
test of that split is to run PPLM-Affinity on the same complexes the MM/GBSA collaborator returns and
compare the two against measurement on identical pairs; this costs almost nothing and settles how much
weight the cheap labels deserve.

Pythia-PPI is accurate on what it does, but it does something else. It needs a complex structure, which
none of the 224,139 candidate pairs has, and it predicts a change caused by a mutation rather than an
absolute affinity, so it cannot label the candidate set under any configuration. Its high agreement with
SKEMPI must also be read with care, because it was trained on SKEMPI, so the figure measures recall of
seen data rather than generalisation. Where it could earn a place is later, generating mutation-level
auxiliary labels around complexes whose structures we already hold, which connects to the auxiliary-target
finding of analysis three. It is not adopted for this block.

Two limitations carry forward. One candidate pair in five contains a protein longer than PPLM-Affinity
accepts and is truncated at 1,022 residues, so those labels come from a partial sequence and should be
flagged in any set we generate. And PPLM-Affinity is released under a non-commercial licence, which suits
academic use but means its weights stay outside the public repository.

## Numbers at a glance

- PPLM-Affinity vs measured pKd, leakage-aware, n 500: Pearson -0.755, Spearman -0.760. Per split: train
  -0.752, validation -0.792, test -0.699. Optimistic all-model figure: -0.845.
- PPLM-Affinity on 200 candidate pairs: range -12.11 to -7.96, spread 4.15, standard deviation 0.62.
- PPLM-Affinity cost: 2.71 s per pair on one GPU in the full setting, 0.27 s in the cheapest; full set
  about 17 hours (one fold model) to 3.5 days (five) on one card; 20.5 s per pair on 8 CPU cores.
- Chain-order averaging changes a prediction by at most 0.000015 and can be dropped.
- Truncation: 40 of 200 candidate pairs exceeded 1,022 residues.
- Pythia-PPI vs measured mutation effects, n 483 over 40 complexes: Pearson +0.897, Spearman +0.892;
  per-complex Pearson median +0.930 (11 complexes with at least 10 measured mutations). Trained on SKEMPI.
- Pythia-PPI cost: about 2 s per structure on one GPU, covering every possible single-point mutation.
- Reference point, first benchmark: the earlier sequence-only teacher scored -0.146 on the same complexes.
- Decision: PPLM-Affinity adopted for soft preheating labels, not as ground truth; MM/GBSA remains the
  source of trustworthy affinity values; Pythia-PPI not adopted for this block.

## Reproduce

From `/home/zaman/Projects/multimodal-ppi` in the esmc environment, submitted through Slurm to a GPU node:

    python phase_c/teacher_pplm_affinity.py --mode benchmark \
      --fasta /home/zaman/Projects/ppi-data/phase_c/candidate_proteins.fasta \
      --pairs /home/zaman/Projects/ppi-data/phase_c/candidate_pairs_clean.csv \
      --limit 200 --device cuda

    python phase_c/teacher_pplm_affinity.py --mode validate \
      --exp_csv /home/zaman/Projects/ppi-data/phase_c/experimental_affinity_pairs.csv \
      --limit 500 --fold_mode heldout --device cuda

    python phase_c/teacher_pythia_ppi.py \
      --skempi /home/zaman/Projects/ppi-data/skempi/skempi_v2.csv \
      --pdb_dir /home/zaman/Projects/ppi-data/skempi/PDBs \
      --n_complexes 40 --device cuda

Outputs: `candidate_pairs_affinity_pplm_sample.csv`, `teacher_validation_pplm_heldout.csv`,
`teacher_validation_pplm_cheap.csv` and `teacher_validation_pythia_ppi.csv`, all under
`~/Projects/ppi-data/phase_c/`. The two models' code and weights live under
`phase_c/teachers/pplm/` and `phase_c/teachers/pythia_ppi/`, both outside git.
