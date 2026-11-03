#!/usr/bin/env bash
# Phase C block one follow-up: identity-threshold leakage audit.
#
# Clusters the unique proteins (phase_c/proteins.fasta, written by the manifest builder)
# with mmseqs2 at a sequence-identity threshold, so that near-duplicate sequences land in
# the same cluster, then runs the analyser to report whether any test protein or test pair
# shares a cluster with a train or validation one, within each source. This catches
# near-duplicates, not just the exact matches the manifest builder already checked. CPU only.
#
# mmseqs2 is already on the cluster (it builds the MSAs). Run from ~/Projects/multimodal-ppi:
#     bash phase_c/leakage_audit.sh                 # defaults: data ~/Projects/ppi-data, id 0.3, cov 0.8
#     bash phase_c/leakage_audit.sh ~/Projects/ppi-data 0.3 0.8
set -euo pipefail

DATA="${1:-$HOME/Projects/ppi-data}"
MINID="${2:-0.3}"
COV="${3:-0.8}"

PC="$DATA/phase_c"
FASTA="$PC/proteins.fasta"
TMP="$PC/mmseqs_tmp"
PREFIX="$PC/clust_id${MINID}"

if [ ! -f "$FASTA" ]; then
  echo "ERROR: $FASTA not found. Run phase_c/build_experimental_manifest.py first." >&2
  exit 1
fi

mkdir -p "$TMP"
echo "clustering $FASTA at min-seq-id $MINID, coverage $COV ..."
mmseqs easy-cluster "$FASTA" "$PREFIX" "$TMP" --min-seq-id "$MINID" -c "$COV" --cov-mode 0 -v 1
echo "cluster table: ${PREFIX}_cluster.tsv"
echo

python phase_c/leakage_audit.py --data "$DATA" --clusters "${PREFIX}_cluster.tsv" --minid "$MINID"

rm -rf "$TMP"
echo
echo "done."
