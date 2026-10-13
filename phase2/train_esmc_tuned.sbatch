#!/usr/bin/env bash
#SBATCH --job-name=esmc_tuned
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=phase2/logs/esmc_tuned_%j.out

# Tuned ESM-C interaction training to raise the headline AUPR/AUROC above the matched baseline:
#   - learning-rate warmup (500 steps) + linear decay
#   - larger effective batch via gradient accumulation (batch 4 x accum 8 = 32)
#   - peak lr 2e-5, 5 epochs; best checkpoint kept by validation AUPR.
# Runs in the dedicated esmc environment.

eval "$(conda shell.bash hook)"; conda activate esmc
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/Projects/multimodal-ppi"
mkdir -p phase2/logs

ESMC=$HOME/Projects/esm-c/ESMplusplus_large
DATA=$HOME/Projects/ppi-data/interaction/bernett
nvidia-smi

python phase2/train_ppi.py --backbone esmc --esm "$ESMC" \
  --train "$DATA/bernett_train.csv" --val "$DATA/bernett_val.csv" \
  --out phase2/runs/esmc_tuned_s0 \
  --epochs 5 --batch 4 --grad_accum 8 --lr 2e-5 --sched linear --warmup_steps 500 \
  --seed 0 --max_length 512

echo "----- ESM-C tuned training done -> phase2/runs/esmc_tuned_s0 -----"
