#!/usr/bin/env bash
# Phase 3 (evo) step 3 - search PPB proteins vs UniRef50 and build one MSA per protein.
#
# Runs mmseqs in BATCHES of queries. The non-batched run OOM'd (the align step tried to
# allocate ~240 GiB); batching keeps peak memory small, and the loop is RESUMABLE - a finished
# batch is skipped on re-run, so an interrupted job just continues.
#
# Usage:  bash phase3/msa_search.sh [QUERY_FASTA] [TAG] [BATCH_SIZE]
#   QUERY_FASTA : proteins to align (default: ~/Projects/ppi-data/ppb_queries.fasta)
#   TAG         : output subfolder ~/Projects/ppi-data/msa/<TAG>   (default: full)
#   BATCH_SIZE  : queries per batch (default: 1000)
# Output: ~/Projects/ppi-data/msa/<TAG>/a3m/<ppbq_id>.a3m   (one MSA per protein)
set -euo pipefail

DATA="$HOME/Projects/ppi-data"
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"

Q="${1:-$DATA/ppb_queries.fasta}"
TAG="${2:-full}"
BATCH="${3:-1000}"
DB="$DATA/uniref50_db"
WORK="$DATA/msa/$TAG"
A3M="$WORK/a3m"
CHUNKS="$WORK/chunks"
THREADS="${SLURM_CPUS_PER_TASK:-16}"
MAXSEQS=1000

mkdir -p "$A3M" "$CHUNKS"
echo "query=$Q  tag=$TAG  batch=$BATCH  threads=$THREADS  max-seqs=$MAXSEQS"

# 1) split the query FASTA into chunks of $BATCH sequences (only once)
if [ -z "$(ls -A "$CHUNKS" 2>/dev/null)" ]; then
  awk -v n="$BATCH" -v d="$CHUNKS" '
    /^>/ { if (c % n == 0) { f++; fn=sprintf("%s/chunk_%04d.fasta", d, f) } c++ }
    { print >> fn }' "$Q"
fi
nchunks=$(ls "$CHUNKS"/chunk_*.fasta | wc -l)
echo "chunks to process: $nchunks"

# 2) process each chunk; skip finished ones (resumable); clean intermediates as we go
i=0
for cf in "$CHUNKS"/chunk_*.fasta; do
  i=$((i+1)); name=$(basename "$cf" .fasta)
  if [ -f "$WORK/$name.done" ]; then echo "[$i/$nchunks] $name already done - skip"; continue; fi
  echo "[$i/$nchunks] $name ..."
  t="$WORK/w_$name"; rm -rf "$t"; mkdir -p "$t"
  mmseqs createdb "$cf" "$t/qdb" > /dev/null
  mmseqs search "$t/qdb" "$DB" "$t/res" "$t/tmp" \
    --threads "$THREADS" -s 5.7 --max-seqs "$MAXSEQS" -e 0.001 --split-memory-limit 100G > /dev/null
  mmseqs result2msa "$t/qdb" "$DB" "$t/res" "$t/msadb" > /dev/null
  mmseqs unpackdb "$t/msadb" "$A3M" --unpack-name-mode 1 --unpack-suffix .a3m > /dev/null
  rm -rf "$t"; touch "$WORK/$name.done"
  echo "    a3m files so far: $(ls "$A3M" | wc -l)"
done

echo "TOTAL a3m files: $(ls "$A3M" | wc -l)  ->  $A3M"
echo "DONE ($TAG)."
