#!/usr/bin/env bash
#SBATCH --job-name=embcov
#SBATCH --partition=gpu_3080ti,gpu_5080,gpu_5070ti,gpu_5090,gpu_6000D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=phase3/logs/embcov_%A.out

# Phase C: sequence-space coverage analysis.
# Embeds the experimental proteins and the synthetic (STRING candidate) proteins with the ESM-C backbone
# (the model's own representation space), then runs the coverage analysis that reports, with numbers and a
# figure, that the synthetic set covers the experimental sequence space and extends beyond it along each
# dimension. Embedding is inference only, so it is light and runs on the usually-free 3080 Ti cards.
eval "$(conda shell.bash hook)"; conda activate esmc
cd "$HOME/Projects/multimodal-ppi"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p phase3/logs
PC="$HOME/Projects/ppi-data/phase_c"

echo "=== embedding experimental proteins ==="
python phase_c/embed_proteins.py --fasta "$PC/proteins.fasta" --out "$PC/emb_experimental"

echo "=== embedding synthetic (candidate) proteins (full set) ==="
python phase_c/embed_proteins.py --fasta "$PC/candidate_proteins.fasta" --out "$PC/emb_synthetic"

echo "=== sequence-space coverage analysis ==="
python phase_c/coverage_analysis.py --exp "$PC/emb_experimental" --syn "$PC/emb_synthetic" --out "$PC"

echo "done. Figure: $PC/coverage_sequence_space.png"
