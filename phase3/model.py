#!/usr/bin/env python
"""
Phase 3 - the multi-task, multimodal model: MultiModalPPI.

Per protein it fuses three views, then reasons across the pair and predicts three things:

  sequence (ESM-2)  --proj-->  + structure (StructEncoder on the contact map)
                               + evolution (ConservationEncoder on MSA features)
                               = per-residue features h  [B, L, d]
  cross-attention (A reads B, B reads A)  ->  z  [B, L, d]
  heads:
    - interaction : do the two proteins bind?           (one number per pair)
    - affinity    : how strongly (pKd)?                  (one number per pair)
    - interface   : which residues form the contact?     (one number per residue)

ALIGNMENT CONTRACT (handled by the data loader, not here): for each protein the token ids
`ids` [B, L], the mask `mask` [B, L], the contact map `adj` [B, L, L] and the evolutionary
features `evo` [B, L, 22] all share the same length L (per batch, padded). Positions that are
padding - and ESM's special tokens - have mask 0, all-zero contact rows, and all-zero evo rows,
so structure/evolution are simply "off" there. This keeps the model itself index-clean.

Structure and evolution are OPTIONAL: pass an all-zero `adj` and/or all-zero `evo` for a protein
that lacks them, and the model falls back to sequence (residual connections make this graceful).

Self-test (CPU, instant - uses a tiny stand-in for ESM):
    python phase3/model.py
"""
import torch
import torch.nn as nn

from struct_module import StructEncoder
from evo_features import ConservationEncoder, FEAT_DIM


class StubSeqEncoder(nn.Module):
    """Tiny stand-in for ESM-2 for CPU unit tests (random per-token embeddings)."""
    def __init__(self, vocab=33, d=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.d = d

    def forward(self, ids, mask):
        return self.emb(ids)


def make_esm_encoder(esm_path):
    """Real sequence encoder: wraps a pretrained ESM-2 (used for actual training)."""
    from transformers import AutoModel

    class EsmSeqEncoder(nn.Module):
        def __init__(self, path):
            super().__init__()
            self.esm = AutoModel.from_pretrained(path)
            self.d = self.esm.config.hidden_size

        def forward(self, ids, mask):
            return self.esm(input_ids=ids, attention_mask=mask).last_hidden_state

    return EsmSeqEncoder(esm_path)


class MultiModalPPI(nn.Module):
    def __init__(self, seq_encoder, d_model=256, n_heads=8, dropout=0.1, struct_layers=2):
        super().__init__()
        self.seq   = seq_encoder
        self.proj  = nn.Linear(seq_encoder.d, d_model)
        self.struct = StructEncoder(d_model, n_layers=struct_layers, dropout=dropout)
        self.evo    = ConservationEncoder(d_model, d_in=FEAT_DIM, dropout=dropout)
        self.a2b = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.b2a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        # heads
        self.iface_head = nn.Linear(d_model, 1)                       # per-residue
        self.inter_head = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.aff_head   = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(d_model, 1))

    def _mean(self, x, mask):
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)

    def encode(self, ids, mask, adj, evo):
        h = self.proj(self.seq(ids, mask))     # sequence -> d_model
        h = self.struct(h, adj)                # + structure (residual inside)
        h = h + self.evo(evo)                  # + evolution
        return h

    def forward(self, a_ids, a_mask, a_adj, a_evo, b_ids, b_mask, b_adj, b_evo):
        hA = self.encode(a_ids, a_mask, a_adj, a_evo)
        hB = self.encode(b_ids, b_mask, b_adj, b_evo)
        a2b, _ = self.a2b(hA, hB, hB, key_padding_mask=(b_mask == 0))   # A reads B
        b2a, _ = self.b2a(hB, hA, hA, key_padding_mask=(a_mask == 0))   # B reads A
        zA = self.norm(hA + a2b)
        zB = self.norm(hB + b2a)
        iface_a = self.iface_head(zA).squeeze(-1)   # [B, La]
        iface_b = self.iface_head(zB).squeeze(-1)   # [B, Lb]
        vA = self._mean(zA, a_mask)
        vB = self._mean(zB, b_mask)
        feats = torch.cat([vA + vB, vA * vB, (vA - vB).abs()], dim=-1)  # symmetric in A,B
        return {
            "interaction": self.inter_head(feats).squeeze(-1),   # [B]
            "affinity":    self.aff_head(feats).squeeze(-1),     # [B]
            "iface_a":     iface_a,                               # [B, La]
            "iface_b":     iface_b,                               # [B, Lb]
        }


# ----------------------------- self-test -----------------------------
def _self_test():
    torch.manual_seed(0)
    B, La, Lb, d_model = 2, 10, 12, 64

    model = MultiModalPPI(StubSeqEncoder(vocab=33, d=32), d_model=d_model, n_heads=4)
    n_params = sum(p.numel() for p in model.parameters())

    def prot(L):
        ids  = torch.randint(0, 33, (B, L))
        mask = torch.ones(B, L)
        mask[0, L - 2:] = 0                          # give sample 0 two padding positions
        adj  = (torch.rand(B, L, L) < 0.1).float()   # sparse random contacts
        adj  = ((adj + adj.transpose(1, 2)) > 0).float()
        for b in range(B):
            adj[b].fill_diagonal_(0)
        evo  = torch.rand(B, L, FEAT_DIM)
        return ids, mask, adj, evo

    a = prot(La)
    b = prot(Lb)
    out = model(*a, *b)

    print(f"params: {n_params:,}")
    for k, v in out.items():
        print(f"  {k:12s} shape {tuple(v.shape)}  finite={torch.isfinite(v).all().item()}")

    # a fake multi-task loss, and confirm gradients flow through everything
    y_int = torch.randint(0, 2, (B,)).float()
    y_aff = torch.rand(B) * 10
    y_ifa = torch.randint(0, 2, (B, La)).float()
    loss = (nn.functional.binary_cross_entropy_with_logits(out["interaction"], y_int)
            + nn.functional.mse_loss(out["affinity"], y_aff)
            + nn.functional.binary_cross_entropy_with_logits(out["iface_a"], y_ifa))
    loss.backward()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"loss: {loss.item():.3f} | gradients flow: {has_grad}")

    assert out["interaction"].shape == (B,)
    assert out["affinity"].shape == (B,)
    assert out["iface_a"].shape == (B, La) and out["iface_b"].shape == (B, Lb)
    assert has_grad
    print("\nOK: MultiModalPPI runs end-to-end (sequence + structure + evolution -> 3 heads).")


if __name__ == "__main__":
    _self_test()
