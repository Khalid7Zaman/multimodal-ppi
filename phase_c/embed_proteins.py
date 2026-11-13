#!/usr/bin/env python
"""
Phase C: per-protein ESM-C embeddings, for the sequence-space coverage analysis.

For each protein in a FASTA, this runs the ESM-C backbone (the model's own representation foundation)
and mean-pools the per-residue features into one vector. The vectors are saved as a NumPy array plus a
matching list of ids, so the coverage analysis can compare the experimental and synthetic proteins in
the same space the model actually uses. Runs on one GPU; it is inference only.

Run from ~/Projects/multimodal-ppi (esmc env):
    python phase_c/embed_proteins.py --fasta <in.fasta> --out <prefix>
writes <prefix>.npy and <prefix>.ids.txt
"""
import os
import argparse
import time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def read_fasta(path):
    ids, seqs, name, buf = [], [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    ids.append(name)
                    seqs.append("".join(buf))
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            ids.append(name)
            seqs.append("".join(buf))
    return ids, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.npy and <out>.ids.txt")
    ap.add_argument("--esm", default=os.path.expanduser("~/Projects/esm-c/ESMplusplus_large"))
    ap.add_argument("--max_length", type=int, default=1022,
                    help="truncate sequences to this many residues for embedding")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.esm, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.esm, trust_remote_code=True,
                                      attn_implementation="sdpa").to(device).eval()

    ids, seqs = read_fasta(args.fasta)
    print(f"{len(ids):,} proteins from {args.fasta}", flush=True)

    # sort by length so each padded batch is tight (faster), remember the original order
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    embs = [None] * len(seqs)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            batch_seqs = [seqs[i][:args.max_length] for i in idx]
            enc = tok(batch_seqs, padding=True, truncation=True, max_length=args.max_length,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**enc).last_hidden_state
            else:
                out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            pooled = pooled.float().cpu().numpy()
            for j, i in enumerate(idx):
                embs[i] = pooled[j]
            done = b + len(idx)
            if (b // args.batch) % 50 == 0 or done == len(seqs):
                print(f"  {done}/{len(seqs)}  ({done / max(time.time() - t0, 1e-6):.0f} prot/s)", flush=True)

    E = np.stack(embs).astype(np.float32)
    np.save(args.out + ".npy", E)
    with open(args.out + ".ids.txt", "w") as f:
        for i in ids:
            f.write(i + "\n")
    print(f"saved {E.shape} -> {args.out}.npy and {args.out}.ids.txt", flush=True)


if __name__ == "__main__":
    main()
