#!/usr/bin/env bash
#SBATCH --job-name=ppi_650M
#SBATCH --partition=gpu_6000D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=phase2/ppi_train_%j.out

eval "$(conda shell.bash hook)"
conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ESM=$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D
DATA=$HOME/Projects/ppi-data/interaction/bernett

nvidia-smi
echo "----- training 650M model -----"
python phase2/train_ppi.py \
  --esm "$ESM" \
  --train "$DATA/bernett_train.csv" \
  --val "$DATA/bernett_val.csv" \
  --epochs 3 --batch 8 --lr 1e-5 --max_length 512 \
  --out phase2/runs/ppi_650M_v1
echo "----- done -----"
