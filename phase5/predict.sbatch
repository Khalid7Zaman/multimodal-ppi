#!/usr/bin/env bash
#SBATCH --job-name=predict
#SBATCH --partition=gpu_5090,gpu_6000D,gpu_5080,gpu_5070ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=phase5/logs/predict_%j.out

# Live demo: predict interaction / affinity / interface for the two sequences in phase5/pair.fasta.
# Put two protein sequences (FASTA, two records) in phase5/pair.fasta, then: sbatch phase5/predict.sbatch
eval "$(conda shell.bash hook)"; conda activate mmppi
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase5/logs
python phase5/predict.py --fasta phase5/pair.fasta
