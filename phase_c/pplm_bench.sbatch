#!/bin/bash
#SBATCH --job-name=pplm_bench
#SBATCH --partition=gpu_5070ti,gpu_3080ti,gpu_3080ti_short,gpu_5080,gpu_5090,gpu_6000D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zaman/Projects/multimodal-ppi/phase_c/pplm_bench_%j.log
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
S=/home/zaman/Projects/multimodal-ppi/phase_c/teacher_pplm_affinity.py
D=/home/zaman/Projects/ppi-data/phase_c
echo "### JOB A: cost and spread on 200 candidate pairs"
python $S --mode benchmark --fasta $D/candidate_proteins.fasta --pairs $D/candidate_pairs_clean.csv --limit 200 --report_every 50 --device cuda --out $D/candidate_pairs_affinity_pplm_sample.csv
echo "### JOB B: cheapest setting, one model and one chain order"
python $S --mode validate --exp_csv $D/experimental_affinity_pairs.csv --limit 500 --fold_mode heldout --no_symmetrize --report_every 100 --device cuda --out $D/teacher_validation_pplm_cheap.csv
