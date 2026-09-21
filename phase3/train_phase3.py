#!/usr/bin/env python
"""
Phase 3 - train the multimodal, multi-task model (MultiModalPPI) on the PPB complexes.

Each complex feeds sequence + structure (contact graph) + evolution (MSA conservation) into the
model, which predicts:
  - binding affinity (pKd)      -> regression, MSE loss
  - interface residues          -> per-residue classification, masked BCE loss (ignores -100)
(The interaction head stays the Phase 2 sequence model; PPB complexes are all interacting, so it
is not trained here.)

Reports: affinity RMSE + Pearson r, and interface AUPR, on the held-out validation split.

Example (SLURM job calls this):
  python phase3/train_phase3.py --esm <path> --epochs 5 --batch 2 --out phase3/runs/mm_v1
  python phase3/train_phase3.py --esm <path> --overfit --limit 16 --epochs 60   # sanity check
"""
import os, argparse, time, math
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score

from model import MultiModalPPI, make_esm_encoder
from data import PPBComplexData, collate, FEAT_DIM  # noqa: F401


def masked_bce(logits, target):
    """BCE over positions with a real label (target >= 0); ignores -100 / start / end / pad."""
    m = target >= 0
    if m.sum() == 0:
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(logits[m], target[m])


def move(b, device):
    out = {}
    for side in ("rec", "lig"):
        out[side] = {k: v.to(device) for k, v in b[side].items()}
    out["pkd"] = b["pkd"].to(device)
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    a_true, a_pred, i_true, i_pred = [], [], [], []
    for b in loader:
        b = move(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["rec"]["ids"], b["rec"]["mask"], b["rec"]["adj"], b["rec"]["evo"],
                        b["lig"]["ids"], b["lig"]["mask"], b["lig"]["adj"], b["lig"]["evo"])
        ok = ~torch.isnan(b["pkd"])
        a_true += b["pkd"][ok].cpu().tolist()
        a_pred += out["affinity"].float()[ok].cpu().tolist()
        for side, key in (("rec", "iface_a"), ("lig", "iface_b")):
            t = b[side]["iface"]; m = t >= 0
            i_true += t[m].cpu().tolist()
            i_pred += torch.sigmoid(out[key].float())[m].cpu().tolist()
    rmse = float(np.sqrt(np.mean((np.array(a_pred) - np.array(a_true)) ** 2))) if a_true else float("nan")
    pear = float(np.corrcoef(a_pred, a_true)[0, 1]) if len(a_true) > 1 else float("nan")
    aupr = float(average_precision_score(i_true, i_pred)) if len(set(i_true)) > 1 else float("nan")
    return rmse, pear, aupr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", required=True)
    ap.add_argument("--out", default="phase3/runs/mm_v1")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--aff_w", type=float, default=1.0)
    ap.add_argument("--iface_w", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overfit", action="store_true", help="train and eval on the same tiny set")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    tok = AutoTokenizer.from_pretrained(args.esm)
    model = MultiModalPPI(make_esm_encoder(args.esm), d_model=256).to(device)

    train_ds = PPBComplexData("train", tok, max_length=args.max_length)
    if args.limit:
        train_ds = Subset(train_ds, list(range(args.limit)))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=2)
    if args.overfit:
        val_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
    else:
        val_dl = DataLoader(PPBComplexData("val", tok, max_length=args.max_length),
                            batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; n = 0; step = 0
        for b in train_dl:
            b = move(b, device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b["rec"]["ids"], b["rec"]["mask"], b["rec"]["adj"], b["rec"]["evo"],
                            b["lig"]["ids"], b["lig"]["mask"], b["lig"]["adj"], b["lig"]["evo"])
                ok = ~torch.isnan(b["pkd"])
                aff = mse(out["affinity"][ok], b["pkd"][ok]) if ok.any() else out["affinity"].sum() * 0.0
                ifc = masked_bce(out["iface_a"], b["rec"]["iface"]) + masked_bce(out["iface_b"], b["lig"]["iface"])
                loss = args.aff_w * aff + args.iface_w * ifc
            loss.backward(); opt.step()
            tot += loss.item() * len(b["pkd"]); n += len(b["pkd"]); step += 1
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step} | loss {loss.item():.4f} "
                      f"(aff {aff.item():.3f} iface {ifc.item():.3f})", flush=True)
        rmse, pear, aupr = evaluate(model, val_dl, device)
        msg = (f"epoch {epoch}: train_loss {tot/max(n,1):.4f} ({time.time()-t0:.0f}s) | "
               f"affinity RMSE {rmse:.3f} Pearson {pear:.3f} | interface AUPR {aupr:.3f}")
        score = (pear if not math.isnan(pear) else -1) + (aupr if not math.isnan(aupr) else 0)
        if score > best:
            best = score; torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            msg += "  [saved best]"
        print(msg, flush=True)
    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
    print("done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
