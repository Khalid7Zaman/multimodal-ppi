#!/bin/bash
S=/home/zaman/Projects/multimodal-ppi/phase_c/teacher_pplm_affinity.py
D=/home/zaman/Projects/ppi-data/phase_c
echo "### JOB 1: cost and spread on candidate pairs"
python $S --mode benchmark --fasta $D/candidate_proteins.fasta --pairs $D/candidate_pairs_clean.csv --limit 60 --report_every 10 --out $D/candidate_pairs_affinity_pplm_sample.csv
echo "### JOB 2: accuracy on measured complexes, leakage-aware"
python $S --mode validate --exp_csv $D/experimental_affinity_pairs.csv --limit 500 --fold_mode heldout --report_every 25
echo "### JOB 3: accuracy the optimistic way, for contrast"
python $S --mode validate --exp_csv $D/experimental_affinity_pairs.csv --limit 500 --fold_mode all --report_every 25
