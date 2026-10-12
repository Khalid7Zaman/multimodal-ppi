#!/usr/bin/env bash
#SBATCH --job-name=train_smoke
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=phase2/logs/train_smoke_%j.out

# Smoke TRAIN: a few hundred pairs for one epoch with each backbone, to confirm the full training
# loop (forward, backward, checkpoint save, evaluation) works for ESM-2 and ESM-C BEFORE the long
# full runs. --overfit evaluates on the same small subset so it stays fast. Runs in the esmc env.

eval "$(conda shell.bash hook)"; conda activate esmc
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/Projects/multimodal-ppi"
mkdir -p phase2/logs

ESM2=$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D
ESMC=$HOME/Projects/esm-c/ESMplusplus_large
DATA=$HOME/Projects/ppi-data/interaction/bernett
nvidia-smi

echo "========== SMOKE TRAIN: ESM-2 =========="
python phase2/train_ppi.py --backbone esm2 --esm "$ESM2" \
  --train "$DATA/bernett_train.csv" \
  --out phase2/runs/smoke_esm2 --overfit --limit 400 --epochs 1 --batch 4 --log_every 20

echo "========== SMOKE TRAIN: ESM-C =========="
python phase2/train_ppi.py --backbone esmc --esm "$ESMC" \
  --train "$DATA/bernett_train.csv" \
  --out phase2/runs/smoke_esmc --overfit --limit 400 --epochs 1 --batch 4 --log_every 20

echo "----- smoke train done (both backbones trained one epoch and saved checkpoints) -----"
