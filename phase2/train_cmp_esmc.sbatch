#!/usr/bin/env bash
#SBATCH --job-name=cmp_esmc
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=phase2/logs/cmp_esmc_%j.out

# Full interaction training with the ESM-C 600M backbone on the Bernett training set, for the
# ESM-2 vs ESM-C comparison. Identical settings to the ESM-2 run (seed 0, 3 epochs, lr 1e-5,
# batch 4, max_length 512). Runs in the dedicated esmc environment.

eval "$(conda shell.bash hook)"; conda activate esmc
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/Projects/multimodal-ppi"
mkdir -p phase2/logs

ESMC=$HOME/Projects/esm-c/ESMplusplus_large
DATA=$HOME/Projects/ppi-data/interaction/bernett
nvidia-smi

python phase2/train_ppi.py --backbone esmc --esm "$ESMC" \
  --train "$DATA/bernett_train.csv" --val "$DATA/bernett_val.csv" \
  --out phase2/runs/cmp_esmc_s0 \
  --epochs 3 --batch 4 --lr 1e-5 --seed 0 --max_length 512

echo "----- ESM-C full training done -> phase2/runs/cmp_esmc_s0 -----"
