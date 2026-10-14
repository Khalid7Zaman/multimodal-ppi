#!/usr/bin/env python
"""
Phase 3 - ESM-C alignment + forward smoke test.

Before training the multimodal model with ESM-C, confirm the per-residue ALIGNMENT CONTRACT that
data.py and model.py depend on still holds with the ESM-C tokenizer:
  - the tokenizer adds exactly one start token (position 0) and one end token (position L-1), so
    residues occupy positions 1..n and the token length L = n + 2 (identical to ESM-2);
  - the structure, evolution and interface tensors line up with the residues, and the special
    positions carry no information (zero contacts, zero conservation, interface label -100).
Then build MultiModalPPI with the real ESM-C encoder and run one finite forward + backward on a
few real PPB complexes, to confirm the model trains end-to-end through the ESM-C backbone.

Runs in the dedicated esmc environment. Usage:
    python phase3/smoke_esmc_mm.py --esm $HOME/Projects/esm-c/ESMplusplus_large --n 4
"""
import os, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model import MultiModalPPI, make_esm_encoder
from data import PPBComplexData, collate, FEAT_DIM  # noqa: F401

# ESM-C load options (same as phase2 / train_phase3.py backbone_load_kwargs("esmc"))
ESM_KWARGS = {"trust_remote_code": True, "attn_implementation": "sdpa"}


def check_alignment(tok):
    """The single highest-risk item: does ESM-C keep residues at positions 1..n?"""
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"          # 33-residue probe
    n = len(seq)
    enc = tok(seq, truncation=True, max_length=512)
    ids = enc["input_ids"]
    L = len(ids)
    interior = tok.decode(ids[1:-1], skip_special_tokens=True).replace(" ", "")
    print(f"alignment probe: residues={n}  tokens={L}  (expect n+2 = {n + 2})")
    print(f"  interior tokens decode back to the sequence: {interior == seq}")
    assert L == n + 2, (
        f"ESM-C produced {L} tokens for {n} residues (adds {L - n} special tokens, not 2). "
        f"data.py places residues at positions 1..n assuming exactly one leading token, so this "
        f"would shift every structure / evolution / interface label. Stop and generalise data.py.")
    print("  PASS: one start token at position 0, residues at 1..n, one end token at L-1.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", required=True)
    ap.add_argument("--n", type=int, default=4, help="how many train complexes to run through")
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.esm, **ESM_KWARGS)
    check_alignment(tok)

    # Assemble a few real complexes and check the per-residue tensors line up.
    ds = PPBComplexData("train", tok, max_length=512, use_msa=True, use_struct=True)
    batch = collate([ds[i] for i in range(args.n)])
    for side in ("rec", "lig"):
        p = batch[side]
        L = p["ids"].shape[1]
        assert p["adj"].shape[1] == L and p["evo"].shape[1] == L and p["iface"].shape[1] == L, \
            "per-residue tensors are not all the same length L"
        # the start token (position 0) must carry no structure / evolution / interface signal
        assert torch.all(p["adj"][:, 0, :] == 0) and torch.all(p["evo"][:, 0] == 0) \
            and torch.all(p["iface"][:, 0] == -100), "special position 0 is not masked off"
        with_struct = (p["adj"].sum(dim=(1, 2)) > 0).sum().item()
        with_evo = (p["evo"].sum(dim=(1, 2)) > 0).sum().item()
        with_if = (p["iface"] >= 0).any(dim=1).sum().item()
        print(f"  {side}: ids{tuple(p['ids'].shape)} adj{tuple(p['adj'].shape)} evo{tuple(p['evo'].shape)}"
              f" | structure {with_struct}/{args.n}  MSA {with_evo}/{args.n}  interface {with_if}/{args.n}")

    print("building MultiModalPPI with the ESM-C encoder ...", flush=True)
    model = MultiModalPPI(make_esm_encoder(args.esm, esm_kwargs=ESM_KWARGS), d_model=256).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  params: {nparams:,}   backbone hidden_size: {model.seq.d}", flush=True)

    def mv(b):
        o = {}
        for s in ("rec", "lig"):
            o[s] = {k: v.to(device) for k, v in b[s].items()}
        o["pkd"] = b["pkd"].to(device)
        return o

    b = mv(batch)
    ok = ~torch.isnan(b["pkd"])
    m = b["rec"]["iface"] >= 0
    # Compute the loss INSIDE autocast, exactly as train_phase3.py does, so the bf16 activations
    # and the loss share a dtype and the backward pass is clean.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(b["rec"]["ids"], b["rec"]["mask"], b["rec"]["adj"], b["rec"]["evo"],
                    b["lig"]["ids"], b["lig"]["mask"], b["lig"]["adj"], b["lig"]["evo"])
        loss = nn.functional.mse_loss(out["affinity"][ok], b["pkd"][ok]) if ok.any() else out["affinity"].sum() * 0.0
        if m.any():
            loss = loss + nn.functional.binary_cross_entropy_with_logits(out["iface_a"][m], b["rec"]["iface"][m])
    for k, v in out.items():
        print(f"  out {k:11s} shape {tuple(v.shape)}  finite={torch.isfinite(v).all().item()}")
    print(f"  (interface labels in this mini-batch: {'yes' if m.any() else 'none - affinity path exercised'})")
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"  loss {float(loss):.4f}   gradients flow: {grad_ok}")
    assert grad_ok, "no finite gradients reached the parameters"

    print("\nSMOKE OK: ESM-C satisfies the alignment contract and the multimodal model trains "
          "end-to-end (sequence + structure + evolution -> affinity + interface).")


if __name__ == "__main__":
    main()
