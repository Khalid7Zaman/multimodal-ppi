#!/usr/bin/env python
"""
Phase 2 - core interaction model: ESM-2 encoder (fine-tuned end-to-end) + cross-attention.
Trains on the Bernett interaction CSVs (query,text,label); reports AUROC / AUPR.
"""
import os, argparse, time, math
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch, torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------- model ----------------
class CrossAttnPPI(nn.Module):
    def __init__(self, esm_path, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.esm = AutoModel.from_pretrained(esm_path)          # ESM-2, fine-tuned end to end
        d = self.esm.config.hidden_size
        self.proj = nn.Linear(d, d_model)
        self.a2b = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.b2a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))

    def _mean(self, x, mask):
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)

    def _encode(self, ids, mask):
        h = self.esm(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.proj(h)

    def forward(self, a_ids, a_mask, b_ids, b_mask):
        hA, hB = self._encode(a_ids, a_mask), self._encode(b_ids, b_mask)
        a2b, _ = self.a2b(hA, hB, hB, key_padding_mask=(b_mask == 0))   # A reads B
        b2a, _ = self.b2a(hB, hA, hA, key_padding_mask=(a_mask == 0))   # B reads A
        vA = self._mean(self.norm(hA + a2b), a_mask)
        vB = self._mean(self.norm(hB + b2a), b_mask)
        feats = torch.cat([vA + vB, vA * vB, (vA - vB).abs()], dim=-1)  # symmetric in A,B
        return self.head(feats).squeeze(-1)

# ---------------- data ----------------
class PairData(Dataset):
    def __init__(self, csv, limit=None):
        self.df = pd.read_csv(csv)
        if limit:
            self.df = self.df.iloc[:limit].reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return str(r["query"]), str(r["text"]), float(r["label"])

def collate_fn(tok, max_length):
    def f(batch):
        A = [b[0] for b in batch]; B = [b[1] for b in batch]
        y = torch.tensor([b[2] for b in batch], dtype=torch.float)
        ta = tok(A, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        tb = tok(B, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return ta.input_ids, ta.attention_mask, tb.input_ids, tb.attention_mask, y
    return f

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, ps = [], []
    for a_ids, a_mask, b_ids, b_mask, y in loader:
        a_ids, a_mask, b_ids, b_mask = (t.to(device) for t in (a_ids, a_mask, b_ids, b_mask))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(a_ids, a_mask, b_ids, b_mask)
        ps += torch.sigmoid(logits.float()).cpu().tolist(); ys += y.tolist()
    au = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    ap = average_precision_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    return au, ap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overfit", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    tok = AutoTokenizer.from_pretrained(args.esm)
    model = CrossAttnPPI(args.esm).to(device)
    coll = collate_fn(tok, args.max_length)

    train_ds = PairData(args.train, limit=args.limit)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=coll, num_workers=2)
    if args.overfit:
        val_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=False, collate_fn=coll)
        print(f"OVERFIT mode on {len(train_ds)} pairs", flush=True)
    elif args.val:
        val_dl = DataLoader(PairData(args.val), batch_size=args.batch, shuffle=False, collate_fn=coll, num_workers=2)
    else:
        val_dl = None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; n = 0
        for a_ids, a_mask, b_ids, b_mask, y in train_dl:
            a_ids, a_mask, b_ids, b_mask, y = (t.to(device) for t in (a_ids, a_mask, b_ids, b_mask, y))
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(a_ids, a_mask, b_ids, b_mask), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        msg = f"epoch {epoch}: train_loss {tot/max(n,1):.4f} ({time.time()-t0:.0f}s)"
        if val_dl is not None:
            au, apr = evaluate(model, val_dl, device)
            msg += f" | val AUROC {au:.4f} AUPR {apr:.4f}"
            if not math.isnan(apr) and apr > best:
                best = apr; torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
        print(msg, flush=True)
    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
    print("done ->", args.out, flush=True)

if __name__ == "__main__":
    main()
