# TriWeave Phase C: distillation teacher benchmark, affinity model (ProteinAffinityKD)

**Author:** Khalid Zaman, Research Assistant Professor (RAP).  **Supervisor:** Prof. Zhaoxi Sun, SUAT.
**Project:** TriWeave (multimodal-ppi).  2026-11-17.

This records the first distillation-teacher benchmark, done before generating any synthetic labels, as the
supervisor asked. The question was whether a released sequence-only binding-affinity model is good enough
and cheap enough to label the candidate pairs. The answer on cost is yes, but on quality it is no, so this
model is not adopted as an affinity teacher. Testing it before trusting it is the point of this step.

## Method

Teacher: ProteinAffinityKD, a released sequence-only protein-protein binding-affinity predictor (a
sequence-based student distilled from a structure-informed teacher), MIT licensed. It outputs a binding
free energy in kcal/mol from sequence alone, using handcrafted Propy features and a small network. The
model is vendored at `phase_c/teachers/proteinaffinitykd/`, run through `phase_c/teacher_affinity_kd.py`.
Features are computed once per unique protein and reused across pairs. CPU only.

Two checks were run. A cost benchmark on a random sample of candidate pairs measured the time per protein
and the spread of the predictions. A validation ran the teacher on 500 of our experimental complexes,
where the binding affinity has been measured, and compared the prediction to the measured pKd. A useful
teacher should show a wide spread of predictions and a strong negative correlation with pKd, since a lower
predicted free energy should mean a higher pKd, that is, stronger binding.

## Result

Cost is low. Feature extraction runs at about 2.7 proteins per second on 8 cores. Labeling the full
candidate set of 44,117 unique proteins would take roughly four to six wall-clock hours on 8 cores,
about 36 to 47 core-hours, with the model forward pass negligible. The cost scales down with more cores or
nodes, and needs no GPU.

Quality is weak, by two independent measures. On a diverse random sample of candidate pairs the predicted
binding free energy spanned only about -11.0 to -10.7 kcal/mol, a range of roughly 0.3, so the teacher
returns nearly the training average for every pair and barely distinguishes one from another. On the 500
experimental complexes with measured affinity, the correlation between the prediction and the measured pKd
was Pearson -0.146 and Spearman -0.168, and near zero on the validation and test portions. The sign is
correct but the strength is far below what a teacher needs, which would be a magnitude above about 0.5.

## What this means

ProteinAffinityKD mostly predicts the mean and cannot reliably rank strong binders against weak ones, on
the candidate proteins or even on measured complexes. Its own documentation notes that it does not reach
the accuracy of structure-based predictors, and this test confirms that concretely on our data. It is
therefore not adopted as an affinity distillation teacher, because distilling from it would teach the model
little more than a constant.

The practical consequence is a cleaner division of the two synthetic-data streams. Binding affinity should
come from the MM/GBSA simulation route, which is physics-based and discriminative, rather than from a weak
sequence-only predictor. Distillation remains useful where strong released sequence models exist, which is
the interaction task rather than affinity. One caveat guides that next step: the current candidate set is
built from STRING high-confidence physical interactions, so every pair is already a positive, and an
interaction teacher run on positives alone would return near-constant high scores for the same reason seen
here. A useful interaction-teacher benchmark therefore has to be validated on a balanced set with real
negatives, such as the leakage-free Bernett benchmark, before any labels are generated.

## Numbers at a glance

- Teacher: ProteinAffinityKD, sequence-only, outputs binding dG (kcal/mol). Vendored, MIT licensed.
- Cost: about 2.7 proteins per second on 8 cores. Full set (44,117 proteins) about 4 to 6 wall-clock hours
  on 8 cores (about 36 to 47 core-hours), CPU only.
- Candidate predictions: range about -11.0 to -10.7 kcal/mol (near-constant, at the training mean -10.8).
- Validation against measured pKd (n 500): Pearson -0.146, Spearman -0.168. Per split: train -0.176,
  validation -0.007, test -0.018.
- Decision: not adopted as an affinity teacher. Affinity to come from MM/GBSA simulation. Distillation to
  be pursued for interaction with a strong sequence model, validated on a balanced benchmark.

## Reproduce

From `~/Projects/multimodal-ppi` in the esmc environment, after `pip install propy3`:

    python phase_c/teacher_affinity_kd.py --mode benchmark \
      --fasta ~/Projects/ppi-data/phase_c/candidate_proteins.fasta \
      --pairs ~/Projects/ppi-data/phase_c/candidate_pairs_clean.csv --limit 300 --workers 8

    python phase_c/teacher_affinity_kd.py --mode validate \
      --exp_csv ~/Projects/ppi-data/phase_c/experimental_affinity_pairs.csv --limit 500 --workers 8
