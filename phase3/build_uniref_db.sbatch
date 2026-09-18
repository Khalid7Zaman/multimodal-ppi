#!/usr/bin/env bash
#SBATCH --job-name=uniref_db
#SBATCH --partition=cpu05_384,cpu03_192,cpu04_192,cpu06_192,cpu02_192_short,cpu01_96_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=phase3/logs/uniref_db_%j.out

# Build the UniRef50 MMseqs2 search database (CPU-only, one-time).
cd "$HOME/Projects/multimodal-ppi"
bash phase3/build_uniref_db.sh
echo "----- done -----"
