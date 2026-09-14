#!/usr/bin/env python
"""
Phase 2 - evaluate a trained CrossAttnPPI checkpoint on a held-out CSV.
Reuses the model/data/eval code from train_ppi.py so the checkpoint loads exactly.
Reports AUROC / AUPR on the test set (the honest, never-seen-before number).
"""
import os, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from train_ppi import CrossAttnPPI, PairData, collate_fn, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    tok = AutoTokenizer.from_pretrained(args.esm)
    model = CrossAttnPPI(args.esm).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    print("loaded checkpoint:", args.ckpt, flush=True)

    coll = collate_fn(tok, args.max_length)
    test_dl = DataLoader(PairData(args.test), batch_size=args.batch,
                         shuffle=False, collate_fn=coll, num_workers=2)
    au, ap_ = evaluate(model, test_dl, device)
    print(f"TEST  AUROC {au:.4f}  AUPR {ap_:.4f}  (n={len(test_dl.dataset)})", flush=True)

if __name__ == "__main__":
    main()
