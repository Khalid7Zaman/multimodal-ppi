#!/usr/bin/env bash
#SBATCH --job-name=ppi_train35M
#SBATCH --partition=gpu_6000D,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=phase2/ppi_train_%j.out

eval "$(conda shell.bash hook)"
conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi
python phase2/train_ppi.py \
  --esm   "$HOME/Projects/PLM-interact/offline/esm2_t12_35M_UR50D" \
  --train "$HOME/Projects/ppi-data/interaction/bernett/bernett_train.csv" \
  --val   "$HOME/Projects/ppi-data/interaction/bernett/bernett_val.csv" \
  --epochs 3 --batch 16 --lr 2e-5 --max_length 512 \
  --out phase2/runs/ppi_35M_v1
echo "----- done -----"
