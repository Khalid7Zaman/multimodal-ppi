#!/usr/bin/env python
"""Per-complex affinity predictions on the held-out TEST split, for the stratified analysis.
Writes a JSON list aligned to ppb_test order: [{"pkd_true":.., "pkd_pred":..}, ...].
The model is evaluated with the SAME view switches it was trained with."""
import os, sys, argparse, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase3"))      # reuse phase3 model/data modules
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import MultiModalPPI, make_esm_encoder
from data import PPBComplexData, collate


def move(b, device):
    out = {}
    for side in ("rec", "lig"):
        out[side] = {k: v.to(device) for k, v in b[side].items()}
    out["pkd"] = b["pkd"].to(device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_struct", action="store_true")
    ap.add_argument("--no_evo", action="store_true")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.esm)
    model = MultiModalPPI(make_esm_encoder(args.esm), d_model=256).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ds = PPBComplexData("test", tok, max_length=args.max_length,
                        use_struct=not args.no_struct, use_msa=not args.no_evo)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=2)

    rows = []
    with torch.no_grad():
        for b in dl:
            b = move(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b["rec"]["ids"], b["rec"]["mask"], b["rec"]["adj"], b["rec"]["evo"],
                            b["lig"]["ids"], b["lig"]["mask"], b["lig"]["adj"], b["lig"]["evo"])
            preds = out["affinity"].float().cpu().tolist()
            trues = b["pkd"].cpu().tolist()
            for t, p in zip(trues, preds):
                rows.append({"pkd_true": t, "pkd_pred": p})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"))
    print(f"wrote {args.out}  ({len(rows)} complexes)  ckpt={args.ckpt}", flush=True)


if __name__ == "__main__":
    main()
