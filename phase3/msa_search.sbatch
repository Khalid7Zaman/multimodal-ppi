#!/usr/bin/env bash
#SBATCH --job-name=ppb_msa
#SBATCH --partition=cpu05_384,cpu03_192,cpu04_192,cpu06_192
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=phase3/logs/ppb_msa_%j.out

# Search PPB proteins vs UniRef50 -> per-protein MSAs, in memory-safe batches.
# Optional args: [QUERY_FASTA] [TAG] [BATCH_SIZE]
#   sbatch phase3/msa_search.sbatch                    # all proteins, batch 1000
#   sbatch phase3/msa_search.sbatch <fasta> test 20    # a tiny test set
cd "$HOME/Projects/multimodal-ppi"
bash phase3/msa_search.sh "${1:-}" "${2:-full}" "${3:-1000}"
echo "----- done -----"
