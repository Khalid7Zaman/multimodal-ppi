#!/usr/bin/env bash
#SBATCH --job-name=eval_tuned
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=phase2/logs/eval_tuned_%j.out

# Score the tuned ESM-C model on the Bernett test set with the full metric set
# (AUPR, AUROC, Precision, Recall, F1, Accuracy, MCC). Inference only. Runs in the esmc env.

eval "$(conda shell.bash hook)"; conda activate esmc
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/Projects/multimodal-ppi"
mkdir -p phase2/logs phase5/metrics_full

ESMC=$HOME/Projects/esm-c/ESMplusplus_large
BTEST=$HOME/Projects/ppi-data/interaction/bernett/bernett_test.csv

python phase2/eval_ppi_full.py --backbone esmc --esm "$ESMC" \
  --ckpt phase2/runs/esmc_tuned_s0/best.pt --test "$BTEST" \
  --out phase5/metrics_full --tag esmc_tuned_s0 --batch 16

echo "----- tuned ESM-C evaluation done -----"
