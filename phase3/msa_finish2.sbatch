#!/usr/bin/env bash
#SBATCH --job-name=ppb_msa2
#SBATCH --partition=gpu_6000D,cpu05_384
#SBATCH --cpus-per-task=2
#SBATCH --mem=360G
#SBATCH --time=2-00:00:00
#SBATCH --output=phase3/logs/ppb_msa2_%j.out

# ---------------------------------------------------------------------------
# FINAL MSA build: result2msa from the SAVED search (res2), the ONE configuration
# proven not to crash - SINGLE-THREADED with ~360 GB on a big-memory node.
# Batching was proven not to reduce the memory (a 500-protein batch still OOM'd at 96 GB),
# so we simply give result2msa the ~240 GB it needs. Search is already done (res2 saved),
# so this is just the last step. 2-day limit so the clock never cuts it off.
# ---------------------------------------------------------------------------
set -euo pipefail
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"
cd "$HOME/Projects/multimodal-ppi"

DATA="$HOME/Projects/ppi-data"
WORK="$DATA/msa/full"
A3M="$WORK/a3m"
CDB="$WORK/hitdb_compact"

mkdir -p "$A3M" phase3/logs
echo "node: $(hostname)   result2msa threads: 1"
echo "--- free memory at start (GB) ---"; free -g

[ -s "$WORK/res2.dbtype" ] || { echo "ERROR: res2 (saved search) missing"; exit 1; }
[ -s "${CDB}.dbtype" ]     || { echo "ERROR: compact DB missing"; exit 1; }

echo "[D] result2msa (single-threaded, from saved res2) - the slow but reliable step ..."
rm -f "$WORK"/msadb2.*
mmseqs result2msa "$WORK/qdb" "$CDB" "$WORK/res2" "$WORK/msadb2" --threads 1

echo "[E] unpack one a3m per protein ..."
mmseqs unpackdb "$WORK/msadb2" "$A3M" --unpack-name-mode 1 --unpack-suffix .a3m

echo "TOTAL a3m files: $(ls "$A3M" | wc -l)  ->  $A3M"
echo "----- done -----"
