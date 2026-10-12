#!/usr/bin/env python
"""
Phase 2 - full-metric evaluation of a trained CrossAttnPPI checkpoint on a held-out CSV.
Reports the same metric set used by the PLM-interact benchmark on the Bernett gold standard:
AUPR, AUROC, Precision, Recall and F1 (Precision/Recall/F1 at the 0.5 threshold), plus Accuracy
and MCC. Saves the per-pair predictions so the metrics and the figure can be rebuilt later
without re-running the model. Reuses the model and data code from train_ppi.py so the checkpoint
loads exactly as in training.

  python phase2/eval_ppi_full.py --esm <ESM_PATH> --ckpt phase2/runs/ppi_650M_v1/best.pt \
      --test <bernett_test.csv> --out phase5/metrics_full --tag ours_650M
"""
import os, argparse, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, accuracy_score, matthews_corrcoef)
from train_ppi import CrossAttnPPI, PairData, collate_fn


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for a_ids, a_mask, b_ids, b_mask, y in loader:
        a_ids, a_mask, b_ids, b_mask = (t.to(device) for t in (a_ids, a_mask, b_ids, b_mask))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(a_ids, a_mask, b_ids, b_mask)
        ps += torch.sigmoid(logits.float()).cpu().tolist()
        ys += y.tolist()
    return ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    tok = AutoTokenizer.from_pretrained(args.esm)
    model = CrossAttnPPI(args.esm).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print("loaded checkpoint:", args.ckpt, flush=True)

    coll = collate_fn(tok, args.max_length)
    dl = DataLoader(PairData(args.test), batch_size=args.batch, shuffle=False,
                    collate_fn=coll, num_workers=2)
    ys, ps = predict(model, dl, device)

    # save predictions so the metrics and figure can be rebuilt without re-running
    pred_csv = os.path.join(args.out, f"preds_{args.tag}.csv")
    pd.DataFrame({"label": [int(y) for y in ys], "prob": ps}).to_csv(pred_csv, index=False)

    yhat = [1 if p >= args.thr else 0 for p in ps]
    m = {
        "tag": args.tag, "n": len(ys), "threshold": args.thr,
        "AUPR":      round(average_precision_score(ys, ps), 4),
        "AUROC":     round(roc_auc_score(ys, ps), 4),
        "Precision": round(precision_score(ys, yhat, zero_division=0), 4),
        "Recall":    round(recall_score(ys, yhat, zero_division=0), 4),
        "F1":        round(f1_score(ys, yhat, zero_division=0), 4),
        "Accuracy":  round(accuracy_score(ys, yhat), 4),
        "MCC":       round(matthews_corrcoef(ys, yhat), 4),
    }
    with open(os.path.join(args.out, f"metrics_{args.tag}.json"), "w") as f:
        json.dump(m, f, indent=2)

    print("=" * 58)
    print(f"  Full metrics on {os.path.basename(args.test)}  (n={m['n']}, threshold {args.thr})")
    for k in ["AUPR", "AUROC", "Precision", "Recall", "F1", "Accuracy", "MCC"]:
        print(f"    {k:10s} {m[k]:.4f}")
    print("=" * 58)
    print("wrote", pred_csv, flush=True)
    print("wrote", os.path.join(args.out, f"metrics_{args.tag}.json"), flush=True)


if __name__ == "__main__":
    main()
