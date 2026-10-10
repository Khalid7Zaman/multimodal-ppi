#!/usr/bin/env python
"""Phase 5 - live demo: two protein sequences -> interaction probability, binding affinity (pKd),
and the top predicted interface residues. Sequence-only inference (no structure / MSA needed;
the model handles their absence). Uses the trained 650M checkpoints.

  python phase5/predict.py --seqA MKT... --seqB MVL...
  python phase5/predict.py --fasta phase5/pair.fasta        # a FASTA with two records
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase3"))
sys.path.insert(0, os.path.join(HERE, "..", "phase2"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoTokenizer
from model import MultiModalPPI, make_esm_encoder    # phase3
from data import FEAT_DIM                            # phase3
from train_ppi import CrossAttnPPI                   # phase2

ESM        = os.path.expanduser("~/Projects/PLM-interact/offline/esm2_t33_650M_UR50D")
INTER_CKPT = os.path.expanduser("~/Projects/multimodal-ppi/phase2/runs/ppi_650M_v1/best.pt")
MM_CKPT    = os.path.expanduser("~/Projects/multimodal-ppi/phase3/runs/mm_650M_s0/best.pt")


def read_fasta(path):
    seqs, cur = [], ""
    for line in open(path):
        if line.startswith(">"):
            if cur:
                seqs.append(cur); cur = ""
        else:
            cur += line.strip()
    if cur:
        seqs.append(cur)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqA"); ap.add_argument("--seqB"); ap.add_argument("--fasta")
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    if args.fasta:
        s = read_fasta(args.fasta)
        assert len(s) >= 2, "FASTA must contain two sequences"
        seqA, seqB = s[0], s[1]
    else:
        seqA, seqB = args.seqA, args.seqB
    assert seqA and seqB, "provide --seqA and --seqB, or --fasta with two records"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ESM)

    def toks(s):
        e = tok(s[:args.max_length - 2], truncation=True, max_length=args.max_length, return_tensors="pt")
        return e.input_ids.to(device), e.attention_mask.to(device).float()

    aid, amask = toks(seqA); bid, bmask = toks(seqB)

    # 1) interaction  (CrossAttnPPI-650M)
    im = CrossAttnPPI(ESM).to(device)
    im.load_state_dict(torch.load(INTER_CKPT, map_location=device)); im.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pinter = torch.sigmoid(im(aid, amask, bid, bmask).float()).item()

    # 2) affinity + interface  (MultiModalPPI-650M; sequence-only: zero structure/evolution)
    mm = MultiModalPPI(make_esm_encoder(ESM), d_model=256).to(device)
    mm.load_state_dict(torch.load(MM_CKPT, map_location=device)); mm.eval()

    def zeros_for(ids):
        L = ids.shape[1]
        return (torch.zeros(1, L, L, device=device), torch.zeros(1, L, FEAT_DIM, device=device))

    aadj, aevo = zeros_for(aid); badj, bevo = zeros_for(bid)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = mm(aid, amask, aadj, aevo, bid, bmask, badj, bevo)
    pkd = out["affinity"].float().item()
    ifa = torch.sigmoid(out["iface_a"][0].float()).cpu().numpy()[1:1 + len(seqA)]
    ifb = torch.sigmoid(out["iface_b"][0].float()).cpu().numpy()[1:1 + len(seqB)]

    def topk(p, seq, k=10):
        idx = np.argsort(-p)[:k]
        return ", ".join(f"{seq[i]}{i + 1}({p[i]:.2f})" for i in sorted(idx))

    print("=" * 64)
    print(f"Protein A: {len(seqA)} residues   |   Protein B: {len(seqB)} residues")
    print(f"Interaction probability          : {pinter:.3f}")
    print(f"Predicted binding affinity (pKd) : {pkd:.2f}")
    print(f"Top predicted interface residues on A : {topk(ifa, seqA)}")
    print(f"Top predicted interface residues on B : {topk(ifb, seqB)}")
    print("=" * 64)
    print("Note: sequence-only inference (no structure/MSA supplied); affinity and interface use the "
          "multimodal model's sequence fallback. Supplying a structure/MSA would use those views too.")


if __name__ == "__main__":
    main()
