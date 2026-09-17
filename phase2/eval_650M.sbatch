#!/usr/bin/env bash
#SBATCH --job-name=ppi_eval650
#SBATCH --partition=gpu_5070ti,gpu_5080,gpu_3080ti,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=phase2/ppi_eval_%j.out

eval "$(conda shell.bash hook)"
conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ESM=$HOME/Projects/PLM-interact/offline/esm2_t33_650M_UR50D
DATA=$HOME/Projects/ppi-data/interaction/bernett

nvidia-smi
echo "----- evaluating 650M best.pt on bernett_test.csv -----"
python phase2/eval_ppi.py \
  --esm "$ESM" \
  --ckpt phase2/runs/ppi_650M_v1/best.pt \
  --test "$DATA/bernett_test.csv" \
  --max_length 512 --batch 8
echo "----- done -----"
