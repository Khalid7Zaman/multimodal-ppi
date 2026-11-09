#!/usr/bin/env bash
# Phase C block two, step 3: remove candidate pairs that leak into the evaluation test set.
#
# Clusters the candidate proteins together with the evaluation test proteins (the clean affinity test
# split plus the Bernett interaction test split) at 30% identity, drops any candidate pair whose
# protein shares a cluster with a test protein, and reports coverage against the targets. CPU only,
# uses mmseqs2 (already on PATH in the esmc env). Run from ~/Projects/multimodal-ppi:
#     bash phase_c/block2_deleak.sh
set -euo pipefail

DATA="${1:-$HOME/Projects/ppi-data}"
PC="$DATA/phase_c"
TMP="$PC/mmseqs_tmp_b2"
PREFIX="$PC/cand_clust_id0.3"

if [ ! -f "$PC/candidate_proteins.fasta" ]; then
  echo "ERROR: $PC/candidate_proteins.fasta not found. Run block2_build_candidates.py first." >&2
  exit 1
fi

echo "prep: building the union fasta (candidate + test proteins) ..."
python3 phase_c/block2_prep_deleak.py --data "$DATA"

echo "clustering candidate + test proteins at 30% identity ..."
mkdir -p "$TMP"
mmseqs easy-cluster "$PC/union_deleak.fasta" "$PREFIX" "$TMP" --min-seq-id 0.3 -c 0.8 --cov-mode 0 -v 1
rm -rf "$TMP"
echo

python3 phase_c/block2_deleak.py --data "$DATA" \
  --clusters "${PREFIX}_cluster.tsv" --test_ids "$PC/eval_test_ids.txt"
echo
echo "done."
