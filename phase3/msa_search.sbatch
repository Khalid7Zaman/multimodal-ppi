#!/usr/bin/env bash
#SBATCH --job-name=ppb_msa
#SBATCH --partition=gpu_6000D,cpu05_384
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=phase3/logs/ppb_msa_%j.out

# ---------------------------------------------------------------------------
# Search PPB proteins vs UniRef50 -> one MSA (.a3m) per protein.
#
# MEMORY LESSON (learned the hard way, 2026-09-21):
#   The mmseqs ALIGNMENT step needs ~258 GB RAM to align against the full
#   58.9M-sequence UniRef50. This is a hard requirement - it is NOT reduced by
#   --split, --split-mode, --split-memory-limit, or --db-load-mode (those only
#   bound the prefilter). So this job MUST run on a big-memory node:
#     gpu05 (gpu_6000D, 515 GB)  or  cpu05_384 (384 GB), with --mem>=320G.
#   Verified: the 20-protein test produced 20 MSAs (~332 homologs each) on gpu05.
#
# Optional args: [QUERY_FASTA] [TAG] [SPLIT]
#   sbatch phase3/msa_search.sbatch                          # all 9,516 proteins
#   sbatch phase3/msa_search.sbatch <fasta> test 16          # a small test set
# ---------------------------------------------------------------------------
cd "$HOME/Projects/multimodal-ppi"
bash phase3/msa_search.sh "${1:-}" "${2:-full}" "${3:-16}"
echo "----- done -----"
