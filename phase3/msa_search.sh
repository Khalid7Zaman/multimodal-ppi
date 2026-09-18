#!/usr/bin/env bash
# Phase 3 (evo) step 3 - search PPB proteins vs UniRef50 and build one MSA per protein.
#
# Usage:  bash phase3/msa_search.sh [QUERY_FASTA] [TAG]
#   QUERY_FASTA : proteins to align (default: ~/Projects/ppi-data/ppb_queries.fasta)
#   TAG         : names the output subfolder ~/Projects/ppi-data/msa/<TAG> (default: full)
#
# Output: ~/Projects/ppi-data/msa/<TAG>/a3m/<ppbq_id>.a3m  (one MSA per protein)
# Needs the UniRef50 DB built first (phase3/build_uniref_db.*). CPU-only, no internet.
set -euo pipefail

DATA="$HOME/Projects/ppi-data"
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"

Q="${1:-$DATA/ppb_queries.fasta}"
TAG="${2:-full}"
DB="$DATA/uniref50_db"
WORK="$DATA/msa/$TAG"
A3M="$WORK/a3m"
THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "$WORK" "$A3M"
echo "query : $Q"
echo "db    : $DB"
echo "work  : $WORK   (threads=$THREADS)"

echo "[1/4] createdb (queries) ..."
mmseqs createdb "$Q" "$WORK/qdb"

echo "[2/4] search vs UniRef50 (heavy step) ..."
mmseqs search "$WORK/qdb" "$DB" "$WORK/res" "$WORK/tmp" \
  --threads "$THREADS" -s 5.7 --max-seqs 2000 -e 0.001 --split-memory-limit 140G

echo "[3/4] result2msa ..."
mmseqs result2msa "$WORK/qdb" "$DB" "$WORK/res" "$WORK/msadb"

echo "[4/4] unpack per-protein a3m files ..."
mmseqs unpackdb "$WORK/msadb" "$A3M" --unpack-name-mode 1 --unpack-suffix .a3m

echo "a3m files written: $(ls "$A3M" | wc -l)  ->  $A3M"
echo "DONE ($TAG)."
