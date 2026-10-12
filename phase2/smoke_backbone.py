#!/usr/bin/env python
"""
Smoke test for a sequence backbone (ESM-2 or ESM-C), run BEFORE any training.
Loads the model and tokenizer, tokenizes two short sequences, runs one forward pass, and prints
the hidden dimension and the output shape. This confirms the model loads and that its interface
(last_hidden_state, attention_mask, hidden size) matches what CrossAttnPPI expects.

  python phase2/smoke_backbone.py --esm <PATH> --backbone esm2
  python phase2/smoke_backbone.py --esm <PATH> --backbone esmc
"""
import os, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from transformers import AutoModel, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", required=True)
    ap.add_argument("--backbone", choices=["esm2", "esmc"], default="esm2")
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    trc = args.backbone == "esmc"
    load_kwargs = {"trust_remote_code": True, "attn_implementation": "sdpa"} if trc else {}

    print(f"backbone = {args.backbone}   path = {args.esm}", flush=True)
    print("loading model ...", flush=True)
    model = AutoModel.from_pretrained(args.esm, **load_kwargs).eval()
    tok = AutoTokenizer.from_pretrained(args.esm, trust_remote_code=trc)

    n_params = sum(p.numel() for p in model.parameters())
    cfg_hidden = getattr(model.config, "hidden_size", None)
    print(f"parameters        : {n_params/1e6:.1f} M")
    print(f"config.hidden_size: {cfg_hidden}")

    seqs = ["MKTAYIAKQRQISFVKSHFSRQ", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"]
    seqs = [s[: args.max_length - 2] for s in seqs]   # leave room for special tokens
    enc = tok(seqs, padding=True, return_tensors="pt")
    print("tokenizer output keys:", list(enc.keys()))
    am = enc.get("attention_mask")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=am)
    lhs = out.last_hidden_state
    print("last_hidden_state shape:", tuple(lhs.shape), "-> hidden dim =", lhs.shape[-1])

    ok = (lhs.dim() == 3) and (cfg_hidden is None or cfg_hidden == lhs.shape[-1])
    print("SMOKE OK" if ok else "SMOKE: check hidden dim vs config")


if __name__ == "__main__":
    main()
