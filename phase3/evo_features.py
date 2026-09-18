#!/usr/bin/env python
"""
Phase 3 - Evolutionary module: turn a protein's MSA into per-residue CONSERVATION features.

Input : an .a3m MSA (from phase3/msa_search.sh) - first sequence is the query, the rest are
        homologs found in UniRef50. In a3m, lowercase letters are insertions relative to the query;
        we drop them so every row lines up with the query's residue positions.
Output: a per-residue feature array [L, 22] =
          - 20 amino-acid frequencies (the position's profile / PSSM),
          - 1 conservation score (1 = fully conserved, 0 = maximally variable),
          - 1 gap fraction.
A protein with no MSA falls back to zeros -> the evolutionary view is simply "uninformative"
for it, and the model still runs (evolution is optional, like structure).

`ConservationEncoder` projects those 22 features to d_model so they fuse with the sequence/structure
features in the full model (built next).

Self-test (no database needed - uses a tiny made-up alignment):
    python phase3/evo_features.py
"""
import os
import numpy as np
import torch
import torch.nn as nn

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
FEAT_DIM = 22  # 20 aa freqs + conservation + gap fraction


def parse_a3m(lines):
    """Parse a3m text (list of lines) -> list of query-aligned sequences (lowercase cols removed)."""
    seqs, buf, have = [], [], False
    for line in lines:
        if line.startswith(">"):
            if have:
                seqs.append("".join(buf))
            buf, have = [], True
        elif have:
            buf.append(line.strip())
    if have:
        seqs.append("".join(buf))
    # drop insertion columns (lowercase); what remains lines up with the query positions
    return ["".join(c for c in s if not c.islower()) for s in seqs]


def msa_to_features(aligned):
    """
    aligned: list of query-aligned sequences (first is the query), all length L.
    Returns (features [L, 22] float32, n_seqs).
    """
    query = aligned[0]
    L = len(query)
    N = len(aligned)
    counts = np.zeros((L, 20), dtype=np.float32)
    gaps = np.zeros(L, dtype=np.float32)
    for s in aligned:
        for j, c in enumerate(s):
            if j >= L:
                break
            if c in AA_IDX:
                counts[j, AA_IDX[c]] += 1.0
            elif c == "-":
                gaps[j] += 1.0
    col_tot = counts.sum(1)
    freq = counts / np.maximum(col_tot[:, None], 1.0)          # [L, 20]
    gap_frac = gaps / max(N, 1)                                # [L]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(freq * np.log(np.clip(freq, 1e-9, 1.0))).sum(1)  # Shannon entropy (nats)
    conservation = 1.0 - ent / np.log(20)                     # 1 = fully conserved
    feats = np.concatenate(
        [freq, conservation[:, None], gap_frac[:, None]], axis=1
    ).astype(np.float32)                                       # [L, 22]
    return feats, N


def features_for_a3m(path):
    """Read an .a3m file and return its [L, 22] features, or None if the file is missing/empty."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path) as f:
        aligned = parse_a3m(f.readlines())
    if not aligned:
        return None
    return msa_to_features(aligned)[0]


class ConservationEncoder(nn.Module):
    """Project per-residue evolutionary features [B, L, 22] -> [B, L, d_model]."""
    def __init__(self, d_model=256, d_in=FEAT_DIM, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_model), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


# ----------------------------- self-test -----------------------------
def _self_test():
    # tiny made-up alignment: position 0 is fully conserved ('A'); the last position varies a lot.
    aligned = [
        "ACDEFGHIKL",  # query
        "ACDEFGHIKM",
        "ACDEFGHIKN",
        "ADDEFGHIKP",  # position 1 also varies a little (C->D once)
        "ACDEFGH-KL",  # a gap at position 7
    ]
    feats, N = msa_to_features(aligned)
    L = len(aligned[0])
    print(f"MSA: {N} sequences, query length {L}, feature dim {feats.shape[1]} (expect {FEAT_DIM})")
    cons = feats[:, 20]
    gap = feats[:, 21]
    print(f"conservation - position 0 (all 'A'): {cons[0]:.2f}  (should be ~1.0)")
    print(f"conservation - last position (L/M/N/P/L): {cons[-1]:.2f}  (should be lower)")
    print(f"gap fraction at position 7: {gap[7]:.2f}  (should be ~0.2 = 1 of 5)")

    # a3m parsing (lowercase insertions must be dropped)
    a3m_lines = [">q", "ACDEF", ">h1", "ACDaEF", ">h2", "AC-EF"]  # h1 has an insertion 'a'; h2 a gap
    parsed = parse_a3m(a3m_lines)
    print(f"a3m parse: query-aligned lengths = {[len(s) for s in parsed]}  (all should equal 5)")

    # encoder shape check
    enc = ConservationEncoder(d_model=256)
    x = torch.from_numpy(feats).unsqueeze(0)          # [1, L, 22]
    out = enc(x)
    print(f"encoder: in {tuple(x.shape)} -> out {tuple(out.shape)}  finite={torch.isfinite(out).all().item()}")

    assert feats.shape[1] == FEAT_DIM
    assert cons[0] > 0.9 and cons[-1] < cons[0]
    assert all(len(s) == 5 for s in parsed)
    assert out.shape == (1, L, 256)
    print("\nOK: evolutionary conservation features + encoder work.")


if __name__ == "__main__":
    _self_test()
