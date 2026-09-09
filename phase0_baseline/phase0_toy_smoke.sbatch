#!/usr/bin/env bash
#SBATCH --job-name=phase0_toy
#SBATCH --partition=gpu_5070ti,gpu_5080,gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=phase0_toy_%j.out

eval "$(conda shell.bash hook)"
conda activate mmppi
cd "$HOME/Projects/PLM-interact"

# Offline mode: read models from local disk, never contact Hugging Face
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nvidia-smi
echo "----- running toy_smoke.py -----"
python toy_smoke.py
echo "----- done -----"
