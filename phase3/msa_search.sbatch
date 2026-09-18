#!/usr/bin/env bash
#SBATCH --job-name=ppb_msa
#SBATCH --partition=cpu05_384,cpu03_192,cpu04_192,cpu06_192
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=phase3/logs/ppb_msa_%j.out

# Search PPB proteins vs UniRef50 -> per-protein MSAs. Pass optional [QUERY_FASTA] [TAG]:
#   sbatch phase3/msa_search.sbatch                         # all proteins (full)
#   sbatch phase3/msa_search.sbatch <fasta> test            # a small test set
cd "$HOME/Projects/multimodal-ppi"
bash phase3/msa_search.sh "${1:-}" "${2:-full}"
echo "----- done -----"
