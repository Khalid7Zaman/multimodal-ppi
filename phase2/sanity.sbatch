#!/usr/bin/env bash
#SBATCH --job-name=ppi_sanity
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=phase2/ppi_sanity_%j.out

eval "$(conda shell.bash hook)"
conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi
python phase2/train_ppi.py \
  --esm  "$HOME/Projects/PLM-interact/offline/esm2_t12_35M_UR50D" \
  --train "$HOME/Projects/ppi-data/interaction/bernett/bernett_tiny64.csv" \
  --overfit --limit 64 --epochs 40 --batch 8 --lr 5e-5 --max_length 512 \
  --out phase2/runs/overfit
echo "----- done -----"
