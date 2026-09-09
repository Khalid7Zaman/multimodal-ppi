#!/usr/bin/env python
import os, time, argparse
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score

HOME = os.path.expanduser("~")
ESM  = f"{HOME}/Projects/PLM-interact/offline/esm2_t33_650M_UR50D"
CKPT = f"{HOME}/Projects/PLM-interact/offline/PLM-interact-650M-humanV11/pytorch_model.bin"

# Same model definition as the toy script that already worked
class PLMinteract(nn.Module):
    def __init__(self, model_name, num_labels, embedding_size):
        super().__init__()
        self.esm_mask = AutoModelForMaskedLM.from_pretrained(model_name)
        self.embedding_size = embedding_size
        self.classifier = nn.Linear(embedding_size, 1)
        self.num_labels = num_labels
    def forward_test(self, features):
        out = self.esm_mask.base_model(**features, return_dict=True)
        emb = F.relu(out.last_hidden_state[:, 0, :])   # CLS token
        logits = self.classifier(emb).view(-1)
        return torch.sigmoid(logits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=1603)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ESM)
    model = PLMinteract(ESM, 1, 1280)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
    model.eval().to(device)

    df = pd.read_csv(args.test)
    n = len(df)
    print(f"loaded {n} pairs from {args.test}", flush=True)

    scores, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, n, args.batch):
            sub = df.iloc[i:i + args.batch]
            tok = tokenizer(list(sub["query"]), list(sub["text"]),
                            padding=True, truncation="longest_first",
                            return_tensors="pt", max_length=args.max_length)
            tok = {k: v.to(device) for k, v in tok.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model.forward_test(tok)
            scores.extend(p.float().cpu().tolist())
            if (i // args.batch) % 50 == 0:
                done = min(i + args.batch, n)
                print(f"  {done}/{n}  ({done/max(time.time()-t0,1e-6):.1f} pairs/s)", flush=True)

    df["score"] = scores
    df.to_csv(args.out, index=False)
    print("wrote", args.out, flush=True)

    if "label" in df.columns and df["label"].nunique() > 1:
        print(f"AUROC = {roc_auc_score(df['label'], df['score']):.4f}", flush=True)
        print(f"AUPR  = {average_precision_score(df['label'], df['score']):.4f}", flush=True)

if __name__ == "__main__":
    main()
