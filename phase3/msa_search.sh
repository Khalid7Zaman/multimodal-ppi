#!/usr/bin/env bash
# Phase 3 (evo) step 3 - search PPB proteins vs UniRef50 -> one MSA (.a3m) per protein.
#
# Memory note: the earlier runs OOM'd with a fixed ~240 GiB allocation regardless of how many
# proteins were searched. Cause: mmseqs sizes its target-DB work from the NODE's total RAM
# (384 GB), not the SLURM --mem limit, so it under-split and the job's memory cap killed it.
# Fix: FORCE the target database to be split into $SPLIT chunks (--split N --split-mode 0),
# which bounds peak memory to roughly (240 GiB / N).
#
# Usage:  bash phase3/msa_search.sh [QUERY_FASTA] [TAG] [SPLIT]
#   QUERY_FASTA : proteins to align (default: ~/Projects/ppi-data/ppb_queries.fasta)
#   TAG         : output subfolder ~/Projects/ppi-data/msa/<TAG>  (default: full)
#   SPLIT       : number of target-DB chunks (default: 16 -> ~15 GiB peak)
set -euo pipefail

DATA="$HOME/Projects/ppi-data"
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"

Q="${1:-$DATA/ppb_queries.fasta}"
TAG="${2:-full}"
SPLIT="${3:-16}"
DB="$DATA/uniref50_db"
WORK="$DATA/msa/$TAG"
A3M="$WORK/a3m"
THREADS="${SLURM_CPUS_PER_TASK:-16}"

rm -rf "$WORK"; mkdir -p "$A3M"
echo "query=$Q  tag=$TAG  split=$SPLIT  threads=$THREADS"

echo "[1/4] createdb (queries) ..."
mmseqs createdb "$Q" "$WORK/qdb"

echo "[2/4] search vs UniRef50 with forced $SPLIT-way target split ..."
mmseqs search "$WORK/qdb" "$DB" "$WORK/res" "$WORK/tmp" \
  --threads "$THREADS" -s 5.7 --max-seqs 1000 -e 0.001 \
  --split "$SPLIT" --split-mode 0

echo "[3/4] result2msa ..."
mmseqs result2msa "$WORK/qdb" "$DB" "$WORK/res" "$WORK/msadb"

echo "[4/4] unpack per-protein a3m files ..."
mmseqs unpackdb "$WORK/msadb" "$A3M" --unpack-name-mode 1 --unpack-suffix .a3m

echo "a3m files: $(ls "$A3M" | wc -l)  ->  $A3M"
echo "DONE ($TAG)."
