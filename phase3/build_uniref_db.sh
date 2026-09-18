#!/usr/bin/env bash
# Phase 3 (evolutionary module) - build the UniRef50 MMseqs2 SEARCH DATABASE.
#
# Input : the downloaded UniRef50 CSV shards in ~/Projects/ppi-data/uniref50_hf/
#         (each row: "index,UniRef50_id,sequence"; line 1 is a stray blank line, skipped).
# Output: ~/Projects/ppi-data/uniref50.fasta  and the MMseqs2 database ~/Projects/ppi-data/uniref50_db
#
# One-time job. No internet needed (runs fine on a CPU compute node).
set -euo pipefail

DATA="$HOME/Projects/ppi-data"
HF="$DATA/uniref50_hf"
FASTA="$DATA/uniref50.fasta"
DB="$DATA/uniref50_db"
export PATH="$HOME/Projects/tools/mmseqs/bin:$PATH"

echo "[1/3] Converting CSV shards -> FASTA ..."
# keep only real data rows (3 columns, id starts with UniRef); strip any stray CR; write FASTA
find "$HF" -name '*.csv' | sort | xargs cat | \
  awk -F',' 'NF>=3 && $2 ~ /^UniRef/ { gsub(/\r/,"",$3); print ">"$2"\n"$3 }' > "$FASTA"
n=$(grep -c '^>' "$FASTA")
echo "    wrote $FASTA  (sequences: $n)"

echo "[2/3] mmseqs createdb ..."
mmseqs createdb "$FASTA" "$DB"

echo "[3/3] Database files:"
ls -lh "$DB"* | awk '{print "    "$5"  "$9}'
echo "DONE. UniRef50 MMseqs2 DB ready at: $DB"
