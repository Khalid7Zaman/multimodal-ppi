#!/usr/bin/env python
"""
Phase 3 - Structural module (part 2): the GRAPH-NETWORK layers.

`struct_features.py` turns a protein structure into a residue contact map A (L x L, 1 = two
residues touch in 3D). This file is the neural network that *uses* that map: a small Graph
Convolutional Network (GCN), written in plain PyTorch (no torch-geometric).

Idea ("message passing"): each residue mixes its features with the features of the residues it
physically contacts. After a couple of layers, every residue's vector reflects its 3D
neighbourhood - the "shape" signal the Phase 2 sequence-only model was missing.

Design choices:
  * Input and output are the SAME shape [B, L, d], so this drops straight into the Phase 2
    pipeline (right after the ESM-2 projection, before cross-attention).
  * A residual connection (output = LayerNorm(H + graph(H))) means that if a protein has no
    usable structure (contact map all zeros), the module falls back to the sequence features -
    structure stays OPTIONAL, as the proposal specifies.

Self-test (CPU, read-only) - runs on a real PPB contact map and random ESM-like features:

    python phase3/struct_module.py
"""
import torch
import torch.nn as nn


def normalize_adj(A):
    """
    Symmetric-normalize a batch of 0/1 contact maps with added self-loops (standard GCN prep).
    A: [B, L, L] float (no self-loops). Returns [B, L, L] = D^(-1/2) (A + I) D^(-1/2).
    """
    L = A.size(-1)
    I = torch.eye(L, device=A.device, dtype=A.dtype).unsqueeze(0)   # [1, L, L]
    A_hat = A + I                                                    # add self-loops
    deg = A_hat.sum(-1)                                             # [B, L]
    d_inv_sqrt = torch.rsqrt(deg.clamp(min=1.0))                    # D^(-1/2), safe if isolated
    # multiply rows and columns by D^(-1/2)
    return A_hat * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)


class GCNLayer(nn.Module):
    """One graph-convolution step: linear transform, then average over 3D neighbours."""
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, A_norm):
        # H: [B, L, d_in]  A_norm: [B, L, L]
        H = self.lin(H)               # transform each residue's features
        H = torch.bmm(A_norm, H)      # aggregate features of contacting residues
        return self.drop(torch.relu(H))


class StructEncoder(nn.Module):
    """
    Structure-aware residue encoder.
      Input : H  [B, L, d]  per-residue features (e.g. ESM-2 projected to d), A [B, L, L] contacts.
      Output:    [B, L, d]  structure-informed per-residue features (same shape).
    """
    def __init__(self, d_model=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [GCNLayer(d_model, d_model, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H, A):
        A_norm = normalize_adj(A)
        X = H
        for layer in self.layers:
            X = layer(X, A_norm)
        return self.norm(H + X)       # residual: structure REFINES the sequence features


# ----------------------------- self-test -----------------------------
def _self_test():
    import numpy as np
    import pandas as pd
    import os
    from struct_features import featurize_complex, PPB  # reuse part 1

    torch.manual_seed(0)
    d_model = 256

    # 1) grab one real, aligned PPB complex and its receptor contact map
    df = pd.read_csv(os.path.join(PPB, "ppb_train_iface.csv"))
    feat = None
    for _, row in df.iterrows():
        f = featurize_complex(row)
        if f is not None and f["receptor"]["aligned"] and f["receptor"]["n_res"] > 10:
            feat = f
            break
    assert feat is not None, "no aligned complex found"
    A_np = feat["receptor"]["A"]                       # [L, L] numpy
    L = A_np.shape[0]
    print(f"Using complex {feat['pdb']} receptor: L = {L} residues, "
          f"{int(A_np.sum()//2)} contacts")

    # 2) fake per-residue features standing in for ESM-2 output (real ESM comes in the full model)
    H = torch.randn(1, L, d_model)                     # [B=1, L, d]
    A = torch.from_numpy(A_np).unsqueeze(0).float()    # [1, L, L]

    enc = StructEncoder(d_model=d_model, n_layers=2)
    n_params = sum(p.numel() for p in enc.parameters())
    out = enc(H, A)

    print(f"input  H: {tuple(H.shape)}")
    print(f"output  : {tuple(out.shape)}   (must match input shape)")
    print(f"params  : {n_params:,}")
    print(f"output finite (no NaN/Inf): {torch.isfinite(out).all().item()}")

    # 3) fallback check: with NO contacts (all-zero map) it must still run and keep sequence info
    out0 = enc(H, torch.zeros_like(A))
    print(f"no-structure fallback runs: {tuple(out0.shape)}, "
          f"finite {torch.isfinite(out0).all().item()}")

    assert out.shape == H.shape and torch.isfinite(out).all()
    print("\nOK: structural graph-network (StructEncoder) works on a real contact map.")


if __name__ == "__main__":
    _self_test()
