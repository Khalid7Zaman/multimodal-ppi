#!/usr/bin/env python
"""
Phase C distillation, teacher 1: run the released ProteinAffinityKD model on our pairs, and benchmark it.

ProteinAffinityKD (Wajid et al., arXiv 2601.03704, MIT licence) is a sequence-only protein-protein
binding-affinity predictor. A structure-informed teacher was distilled into a sequence-based student at
training time, so at inference it needs sequences only. It outputs a binding free energy in kcal/mol (more
negative means stronger binding). The vendored model lives in phase_c/teachers/proteinaffinitykd/.

This wrapper does three things, controlled by --mode:
  benchmark : run on a sample of candidate pairs, time it, and extrapolate the cost to the full set
              (this answers the supervisor's request to benchmark the cost per calculation first).
  pairs     : run on all candidate pairs and write a predicted-affinity label per pair (the distillation
              labels), reusing per-protein features so each protein is featurised once.
  validate  : run on our experimental affinity pairs and correlate the teacher's prediction with the
              measured pKd, so we can judge how trustworthy the teacher's labels are before distilling.

The cost is dominated by the Propy feature extraction, about one second per protein, and features are
computed once per unique protein and cached, so the model forward pass is negligible. CPU only.

Run from ~/Projects/multimodal-ppi (esmc env, after `pip install propy3`):
  python phase_c/teacher_affinity_kd.py --mode benchmark \
      --fasta ~/Projects/ppi-data/phase_c/candidate_proteins.fasta \
      --pairs ~/Projects/ppi-data/phase_c/candidate_pairs_clean.csv --limit 300 --workers 8
  python phase_c/teacher_affinity_kd.py --mode validate \
      --exp_csv ~/Projects/ppi-data/phase_c/experimental_affinity_pairs.csv --workers 8
"""
import os
import time
import argparse
import hashlib
import random
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "teachers", "proteinaffinitykd")
AA = list("ARNDCEQGHILKMFPSTWYV")
_KEYLIST = None


def load_keylist(model_dir):
    global _KEYLIST
    if _KEYLIST is None:
        k = np.load(os.path.join(model_dir, "key_list_proPy.npy"))
        _KEYLIST = np.char.decode(k, encoding="utf-8")
    return _KEYLIST


def featurise(seq, model_dir, seed=0):
    """Propy descriptors for one sequence, selecting the model's fixed key list. 1537-vector."""
    from propy.PyPro import GetProDes
    keylist = load_keylist(model_dir)
    rng = random.Random(seed)
    s = "".join(ch for ch in seq.upper() if ch.isalpha())
    for bad in "XBUZ":
        while bad in s:
            s = s.replace(bad, AA[rng.randrange(len(AA))], 1)
    try:
        alld = GetProDes(s).GetALL()
    except Exception:
        return np.zeros(len(keylist), dtype=np.float32)
    return np.array([alld.get(k, 0.0) for k in keylist], dtype=np.float32)


def _feat_worker(args):
    seq, model_dir = args
    return featurise(seq, model_dir)


def featurise_unique(seqs, model_dir, workers):
    """Featurise each distinct sequence once. Returns {seq: vector} and the elapsed time."""
    uniq = sorted(set(seqs))
    t0 = time.time()
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            feats = pool.map(_feat_worker, [(s, model_dir) for s in uniq], chunksize=4)
    else:
        feats = [featurise(s, model_dir) for s in uniq]
    dt = time.time() - t0
    return dict(zip(uniq, feats)), dt, len(uniq)


def load_model(model_dir):
    import torch
    import torch.nn as nn
    import joblib

    class StudentNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3074, 384)
            self.fc2 = nn.Linear(384, 128)
            self.fc3 = nn.Linear(128, 16)
            self.out = nn.Linear(16, 1)
            self.drop1 = nn.Dropout(0.3)
            self.drop2 = nn.Dropout(0.2)

        def forward(self, x):
            h = torch.relu(self.fc1(x)); h = self.drop1(h)
            h = torch.relu(self.fc2(h)); h = self.drop2(h)
            f = torch.relu(self.fc3(h))
            return self.out(f), f

    model = StudentNet()
    model.load_state_dict(torch.load(os.path.join(model_dir, "student_with_diss_weights.pth"),
                                     map_location="cpu"))
    model.eval()
    sc = joblib.load(os.path.join(model_dir, "scaler_propy.pkl"))
    ls = joblib.load(os.path.join(model_dir, "scaler_labels.pkl"))
    return model, sc, ls


def predict(jobs, feat, model_dir, symmetrize=False):
    """jobs: list of (id_a, seq_a, id_b, seq_b). Returns predicted binding dG (kcal/mol) per job."""
    import torch
    from sklearn.preprocessing import normalize
    model, sc, ls = load_model(model_dir)

    def stack(order_ab):
        X = []
        for (ia, sa, ib, sb) in jobs:
            a, b = (sa, sb) if order_ab else (sb, sa)
            X.append(np.concatenate((feat[a], feat[b])))
        Xs = normalize(sc.transform(np.asarray(X, dtype=np.float32)))
        with torch.no_grad():
            out, _ = model(torch.tensor(Xs, dtype=torch.float32))
        return ls.inverse_transform(out.numpy()).ravel()

    pred = stack(True)
    if symmetrize:
        pred = (pred + stack(False)) / 2.0
    return pred


def read_fasta(path):
    d, name, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                d[name] = "".join(buf)
            name = line[1:].split()[0]; buf = []
        else:
            buf.append(line)
    if name:
        d[name] = "".join(buf)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["benchmark", "pairs", "validate"], required=True)
    ap.add_argument("--model_dir", default=DEFAULT_MODEL)
    ap.add_argument("--fasta", help="protein FASTA (ids = headers), for benchmark/pairs modes")
    ap.add_argument("--pairs", help="candidate pairs CSV (protein1_id, protein2_id), for benchmark/pairs")
    ap.add_argument("--exp_csv", help="experimental_affinity_pairs.csv, for validate mode")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="benchmark: use only the first N pairs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--symmetrize", action="store_true",
                    help="predict both chain orders and average (robust to the ligand/receptor order)")
    args = ap.parse_args()

    print("=" * 80)
    print(f"Phase C distillation teacher: ProteinAffinityKD  (mode: {args.mode})")
    print("=" * 80)

    if args.mode in ("benchmark", "pairs"):
        seqs = read_fasta(args.fasta)
        pairs = pd.read_csv(args.pairs, dtype={"protein1_id": str, "protein2_id": str})
        if args.mode == "benchmark" and args.limit and len(pairs) > args.limit:
            # random (seeded) sample, so timing and the spread of predictions are representative
            pairs = pairs.sample(n=args.limit, random_state=0).reset_index(drop=True)
        jobs = []
        miss = 0
        for _, r in pairs.iterrows():
            a, b = r["protein1_id"], r["protein2_id"]
            if a in seqs and b in seqs:
                jobs.append((a, seqs[a], b, seqs[b]))
            else:
                miss += 1
        needed = [s for (_, s, _, s2) in jobs for s in (s, s2)]
        print(f"pairs: {len(jobs):,} usable ({miss:,} skipped for a missing sequence)")
        feat, ft, nu = featurise_unique(needed, args.model_dir, args.workers)
        print(f"featurised {nu:,} unique proteins in {ft:.1f}s "
              f"({ft / max(nu, 1):.3f}s per protein, {args.workers} workers)")
        t0 = time.time()
        pred = predict(jobs, feat, args.model_dir, symmetrize=args.symmetrize)
        pt = time.time() - t0
        print(f"predicted {len(jobs):,} pairs in {pt:.2f}s")
        print(f"predicted binding dG (kcal/mol): min {pred.min():.2f}, median {np.median(pred):.2f}, "
              f"max {pred.max():.2f}")
        out = args.out or os.path.join(os.path.dirname(args.pairs), "candidate_pairs_affinity.csv")
        res = pairs.iloc[:len(jobs)].copy()
        res = pd.DataFrame({"protein1_id": [j[0] for j in jobs], "protein2_id": [j[2] for j in jobs],
                            "pred_dG_kcalmol": pred})
        res.to_csv(out, index=False)
        print("wrote:", out)

        if args.mode == "benchmark":
            allp = pd.read_csv(args.pairs, dtype={"protein1_id": str, "protein2_id": str})
            all_prot = set(allp["protein1_id"]) | set(allp["protein2_id"])
            per_wall = ft / max(nu, 1)                   # wall seconds per protein at `workers`
            wall_h = per_wall * len(all_prot) / 3600.0   # wall-clock hours at `workers`
            core_h = wall_h * max(args.workers, 1)
            print("\n--- cost extrapolation to the full candidate set ---")
            print(f"  full set: {len(allp):,} pairs over {len(all_prot):,} unique proteins")
            print(f"  feature throughput: {max(nu,1)/ft:.2f} proteins/s at {args.workers} workers "
                  f"({per_wall:.3f} wall-s/protein); the model forward pass is negligible")
            print(f"  full set -> about {wall_h:.1f} wall-clock hours at {args.workers} workers "
                  f"(~{core_h:.0f} core-hours). More cores or nodes cut the wall time proportionally.")
        return

    # validate: teacher prediction vs measured pKd on the experimental pairs
    from scipy import stats
    df = pd.read_csv(args.exp_csv)
    df = df[pd.to_numeric(df["pKd"], errors="coerce").notna()].copy()
    df["pKd"] = df["pKd"].astype(float)
    if args.limit and len(df) > args.limit:
        df = df.sample(n=args.limit, random_state=0).reset_index(drop=True)
    jobs = [(str(i) + "L", r["seq_ligand"], str(i) + "R", r["seq_receptor"])
            for i, r in df.iterrows()]
    needed = [s for (_, s, _, s2) in jobs for s in (s, s2)]
    feat, ft, nu = featurise_unique(needed, args.model_dir, args.workers)
    print(f"featurised {nu:,} unique sequences in {ft:.1f}s ({ft/max(nu,1):.3f}s each)")
    pred = predict(jobs, feat, args.model_dir, symmetrize=args.symmetrize)
    df["pred_dG_kcalmol"] = pred
    m = np.isfinite(df["pred_dG_kcalmol"]) & np.isfinite(df["pKd"])
    pr = stats.pearsonr(df["pred_dG_kcalmol"][m], df["pKd"][m])
    srho = stats.spearmanr(df["pred_dG_kcalmol"][m], df["pKd"][m])
    print(f"\nteacher dG vs measured pKd  (expect NEGATIVE: lower dG = higher pKd = stronger)")
    print(f"  n = {int(m.sum()):,}   Pearson r = {pr[0]:.3f}   Spearman r = {srho[0]:.3f}")
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s] if "split" in df.columns else df.iloc[0:0]
        mm = np.isfinite(sub["pred_dG_kcalmol"]) & np.isfinite(sub["pKd"])
        if mm.sum() >= 10:
            r = stats.pearsonr(sub["pred_dG_kcalmol"][mm], sub["pKd"][mm])[0]
            print(f"  split {s:>5}: n {int(mm.sum()):>5,}  Pearson {r:.3f}")
    out = args.out or os.path.join(os.path.dirname(args.exp_csv), "teacher_validation_affinitykd.csv")
    df.to_csv(out, index=False)
    print("wrote:", out)
    print("\nNOTE: ProteinAffinityKD was trained on an affinity benchmark that may overlap our PPB set,"
          " so this is an optimistic check, strongest as a sanity test of scale and sign.")


if __name__ == "__main__":
    main()
