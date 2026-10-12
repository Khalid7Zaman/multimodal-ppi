#!/usr/bin/env bash
#SBATCH --job-name=cmp_esm2
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=phase2/logs/cmp_esm2_%j.out

# Full interaction training with the ESM-2 backbone on the Bernett training set, for the
# ESM-2 vs ESM-C comparison. Identical settings to the ESM-C run (seed 0, 3 epochs, lr 1e-5,
# batch 4, max_length 512). Runs in the dedicated esmc environment.

eval "$(conda shell.bash hook)"; conda activate esmc
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/Projects/multimodal-ppi"
mkdir -p phase2/logs

ESM2=$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D
DATA=$HOME/Projects/ppi-data/interaction/bernett
nvidia-smi

python phase2/train_ppi.py --backbone esm2 --esm "$ESM2" \
  --train "$DATA/bernett_train.csv" --val "$DATA/bernett_val.csv" \
  --out phase2/runs/cmp_esm2_s0 \
  --epochs 3 --batch 4 --lr 1e-5 --seed 0 --max_length 512

echo "----- ESM-2 full training done -> phase2/runs/cmp_esm2_s0 -----"
