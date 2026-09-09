#!/usr/bin/env bash
#SBATCH --job-name=phase0_bench
#SBATCH --partition=gpu_5090,gpu_5080,gpu_5070ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=phase0_bench_%j.out

eval "$(conda shell.bash hook)"
conda activate mmppi
cd "$HOME/Projects/PLM-interact"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CSV=${1:-human_test_small.csv}      # pass a CSV name as the 1st argument
OUT=${CSV%.csv}_scored.csv
echo "input: $CSV   output: $OUT"
nvidia-smi
python benchmark_infer.py --test "$CSV" --out "$OUT" --batch 8
echo "----- done -----"
