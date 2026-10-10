#!/usr/bin/env python
"""Phase 5 Part C - extract interpretability data for a few example PPB TEST complexes:
cross-attention weights (receptor->ligand), per-residue interface probabilities, C-alpha
coordinates, and true interface labels. Saves one small .npz per example to phase5/interp/.

Picks the examples automatically: among complexes whose two chains both align 1:1 to the
structure and are not too long, keep the K with the best predicted interface (so the figures
are clear, illustrative cases). Uses the definitive additive 650M checkpoint."""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase3"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score
from model import MultiModalPPI, make_esm_encoder
from data import PPBComplexData, collate
from struct_features import featurize_complex


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
    ap.add_argument("--k", type=int, default=3, help="how many top examples to save")
    ap.add_argument("--max_res", type=int, default=400, help="skip chains longer than this (clean alignment, no truncation)")
    ap.add_argument("--outdir", default="phase5/interp")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.esm)
    model = MultiModalPPI(make_esm_encoder(args.esm), d_model=256).to(device)   # additive (default)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ds = PPBComplexData("test", tok, max_length=512, use_struct=True, use_msa=True)

    # 1) candidates: both chains aligned 1:1 and short enough to avoid truncation
    cand = []
    for i in range(len(ds)):
        row = ds.df.iloc[i]
        if len(str(row["query"])) > args.max_res or len(str(row["text"])) > args.max_res:
            continue
        feat = featurize_complex(row)
        if feat is None or not (feat["receptor"]["aligned"] and feat["ligand"]["aligned"]):
            continue
        cand.append(i)
    print(f"{len(cand)} candidate complexes (both chains aligned, <= {args.max_res} residues)", flush=True)

    # 2) rank candidates by interface AUPR
    scored = []
    for i in cand:
        b = move(collate([ds[i]]), device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["rec"]["ids"], b["rec"]["mask"], b["rec"]["adj"], b["rec"]["evo"],
                        b["lig"]["ids"], b["lig"]["mask"], b["lig"]["adj"], b["lig"]["evo"])
        yt, yp = [], []
        for side, key in (("rec", "iface_a"), ("lig", "iface_b")):
            t = b[side]["iface"][0]; m = t >= 0
            yt += t[m].cpu().tolist(); yp += torch.sigmoid(out[key][0].float())[m].cpu().tolist()
        if len(set(yt)) > 1:
            scored.append((average_precision_score(yt, yp), i))
    scored.sort(reverse=True)
    picks = scored[:args.k]
    print("picked (interface AUPR, pdb):", [(round(s, 3), ds.df.iloc[i]["pdb"]) for s, i in picks], flush=True)

    # 3) for each pick, capture attention + probs + coords + labels
    for aupr, i in picks:
        row = ds.df.iloc[i]
        b = move(collate([ds[i]]), device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hA = model.encode(b["rec"]["ids"], b["rec"]["mask"], b["rec"]["adj"], b["rec"]["evo"])
            hB = model.encode(b["lig"]["ids"], b["lig"]["mask"], b["lig"]["adj"], b["lig"]["evo"])
            a2b_out, a2b_w = model.a2b(hA, hB, hB, key_padding_mask=(b["lig"]["mask"] == 0),
                                       need_weights=True, average_attn_weights=True)
            b2a_out, _ = model.b2a(hB, hA, hA, key_padding_mask=(b["rec"]["mask"] == 0),
                                   need_weights=False)
            zA = model.norm(hA + a2b_out); zB = model.norm(hB + b2a_out)
            pa = torch.sigmoid(model.iface_head(zA)[0, :, 0].float()).cpu().numpy()
            pb = torch.sigmoid(model.iface_head(zB)[0, :, 0].float()).cpu().numpy()
        feat = featurize_complex(row)
        nr, nl = feat["receptor"]["n_res"], feat["ligand"]["n_res"]
        # residues occupy token positions 1..n (position 0 is the start token)
        np.savez(os.path.join(args.outdir, f"{feat['pdb']}.npz"),
                 pdb=feat["pdb"], aupr=float(aupr),
                 rec_coords=feat["receptor"]["coords"][:nr], lig_coords=feat["ligand"]["coords"][:nl],
                 rec_prob=pa[1:1 + nr], lig_prob=pb[1:1 + nl],
                 rec_true=b["rec"]["iface"][0, 1:1 + nr].cpu().numpy(),
                 lig_true=b["lig"]["iface"][0, 1:1 + nl].cpu().numpy(),
                 a2b=a2b_w[0, 1:1 + nr, 1:1 + nl].float().cpu().numpy())
        print(f"saved {feat['pdb']}.npz  (receptor {nr} res, ligand {nl} res, interface AUPR {aupr:.3f})", flush=True)


if __name__ == "__main__":
    main()
