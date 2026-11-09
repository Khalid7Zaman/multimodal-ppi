#!/usr/bin/env python
"""
Phase C block two, step 2: build the candidate interacting-pair pool from the STRING downloads.

It reads the per-organism STRING files in <data>/string/ (fetched by block2_fetch_string.sh), keeps
the high-confidence PHYSICAL interactions as plausible binders, which are the positive pairs for
synthetic affinity generation, attaches the protein sequences, pools across organisms, removes
duplicates, and writes:
  <data>/phase_c/candidate_pairs.csv       pair_key, taxid, protein1, protein2, combined_score, len1, len2
  <data>/phase_c/candidate_proteins.fasta  the unique candidate proteins
Block two step 3 then removes any pair too similar to the experimental test set and samples down to
the size and coverage targets. This step only reads and writes files, no GPU.

Run from ~/Projects/multimodal-ppi:
    python phase_c/block2_build_candidates.py                 # min combined_score 700 (high confidence)
    python phase_c/block2_build_candidates.py --min_score 900 # highest confidence only
"""
import os
import gzip
import glob
import argparse
import hashlib
import pandas as pd


def sid(seq, n=12):
    return hashlib.sha1(seq.encode()).hexdigest()[:n]


def clean_seq(s):
    return "".join(s.split()).upper()


def read_sequences(path):
    """FASTA(.gz) -> {string_id: sequence}. Headers look like >511145.b0001."""
    seqs, name, buf = {}, None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            seqs[name] = "".join(buf)
    return seqs


def read_links(path, min_score):
    """STRING physical links(.gz) -> list of (protein1, protein2, score) with score >= min_score.

    Works for both the plain file (protein1 protein2 combined_score) and the 'full' file (many
    evidence columns then combined_score last), because the score is always the LAST column. A text
    header line is detected and skipped by checking whether the last token is a number.
    """
    pairs = []
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                score = int(parts[-1])
            except ValueError:
                continue  # header or malformed line
            if score >= min_score:
                pairs.append((parts[0], parts[1], score))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Projects/ppi-data"))
    ap.add_argument("--min_score", type=int, default=700,
                    help="minimum STRING combined_score (0-1000); 700 = high, 900 = highest")
    args = ap.parse_args()

    data = os.path.expanduser(args.data)
    sdir = os.path.join(data, "string")
    pc = os.path.join(data, "phase_c")
    os.makedirs(pc, exist_ok=True)

    link_files = sorted(glob.glob(os.path.join(sdir, "*.protein.physical.links.*.txt.gz")))
    if not link_files:
        raise SystemExit(f"No STRING link files in {sdir}. Run block2_fetch_string.sh first.")

    print("=" * 70)
    print(f"Phase C block two step 2: candidate pairs (min combined_score {args.min_score})")
    print("=" * 70)

    rows = []
    seq_of_id = {}       # candidate protein id -> sequence
    per_org = []
    for lf in link_files:
        taxid = os.path.basename(lf).split(".")[0]
        sf = os.path.join(sdir, f"{taxid}.protein.sequences.v12.0.fa.gz")
        if not os.path.exists(sf):
            print(f"  {taxid}: no sequences file, skipped")
            continue
        seqs = read_sequences(sf)
        pairs = read_links(lf, args.min_score)
        kept = 0
        for p1, p2, score in pairs:
            s1, s2 = seqs.get(p1), seqs.get(p2)
            if not s1 or not s2:
                continue
            a, b = clean_seq(s1), clean_seq(s2)
            if not a or not b:
                continue
            ida, idb = sid(a), sid(b)
            key = (ida, idb) if ida <= idb else (idb, ida)
            rows.append({
                "pair_key": key[0] + "_" + key[1],
                "taxid": taxid,
                "protein1": p1, "protein2": p2,
                "protein1_id": ida, "protein2_id": idb,
                "combined_score": score,
                "len1": len(a), "len2": len(b),
            })
            seq_of_id.setdefault(ida, a)
            seq_of_id.setdefault(idb, b)
            kept += 1
        per_org.append((taxid, len(pairs), kept))
        print(f"  {taxid}: {len(pairs):>8d} high-confidence edges, {kept:>8d} with both sequences")

    if not rows:
        raise SystemExit("No candidate pairs built. Check the downloads and --min_score.")

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset=["pair_key"]).reset_index(drop=True)
    dups = before - len(df)

    pairs_path = os.path.join(pc, "candidate_pairs.csv")
    df[["pair_key", "taxid", "protein1", "protein2", "protein1_id", "protein2_id",
        "combined_score", "len1", "len2"]].to_csv(pairs_path, index=False)

    fasta_path = os.path.join(pc, "candidate_proteins.fasta")
    with open(fasta_path, "w") as f:
        for pid, seq in seq_of_id.items():
            f.write(f">{pid}\n{seq}\n")

    print()
    print(f"candidate pairs: {len(df):,} (removed {dups:,} cross-organism duplicate pairs)")
    print(f"unique candidate proteins: {len(seq_of_id):,}")
    print(f"score range kept: {int(df.combined_score.min())}-{int(df.combined_score.max())}")
    print()
    print("Wrote:")
    print("  ", pairs_path)
    print("  ", fasta_path)
    print()
    print("Next: block two step 3 removes pairs too similar to the experimental test set and samples")
    print("to the target of at least 80,450 pairs with broad coverage.")


if __name__ == "__main__":
    main()
