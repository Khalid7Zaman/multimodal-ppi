#!/usr/bin/env python
"""
Phase C distillation, teacher 2: benchmark the released PPLM-Affinity model on our pairs.

PPLM-Affinity is a paired-sequence protein language model (650M-parameter backbone, 33 blocks) with a
binding-affinity head, released by its authors with five cross-validation models. It takes two sequences
only, no structure, and outputs one binding free energy per pair, so unlike structure-based predictors it
can in principle label our STRING candidate pairs. The released weights file holds all five fold models.

The same two questions as the first teacher benchmark, so the results are directly comparable:
  benchmark : run on a random sample of candidate pairs, time it, extrapolate the cost to the full set,
              and report the spread of the predictions (a teacher that returns a near-constant value
              teaches nothing, which is how the first teacher failed).
  validate  : run on our experimental complexes, where affinity has been measured, and correlate the
              prediction with the measured pKd.
  pairs     : label a set of candidate pairs and write one predicted value per pair.

LEAKAGE, and why this script does not simply average the five models. PPLM-Affinity was fine-tuned on
PPB-Affinity, which is the source of our own experimental affinity set, so most of our complexes were in
its training data. Averaging all five fold models over such a complex measures memorisation, not skill.
The authors publish the PDB entries of each fold (data/affinity/cv_folds.csv), so validate mode supports:
  --fold_mode heldout : for a complex in fold k, predict with fold model k only, which is the model that
                        did not train on that fold. This is the honest estimate.
  --fold_mode unseen  : restrict to complexes that appear in no fold at all, and average all five models.
                        The strictest check, limited by how many such complexes we have.
  --fold_mode all     : average all five models over everything. Optimistic, reported only for contrast.
Candidate-pair labelling uses the five-model average, as the authors intend, since those pairs are not in
the affinity training set.

Both chain orders are averaged by default (--no_symmetrize turns that off), as the released script does.

Run from /home/zaman/Projects/multimodal-ppi (esmc env):
  python phase_c/teacher_pplm_affinity.py --mode benchmark \
      --fasta ~/Projects/ppi-data/phase_c/candidate_proteins.fasta \
      --pairs ~/Projects/ppi-data/phase_c/candidate_pairs_clean.csv --limit 50
  python phase_c/teacher_pplm_affinity.py --mode validate \
      --exp_csv ~/Projects/ppi-data/phase_c/experimental_affinity_pairs.csv \
      --limit 500 --fold_mode heldout
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CODE = os.path.join(HERE, "teachers", "pplm", "pplm_code")
DEFAULT_WEIGHTS = os.path.join(HERE, "teachers", "pplm", "affinity_models.pkl")


def read_fasta(path):
    d, name, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                d[name] = "".join(buf)
            name = line[1:].split()[0]
            buf = []
        else:
            buf.append(line)
    if name:
        d[name] = "".join(buf)
    return d


def clean_seq(seq, max_len):
    """Keep letters only, map the residue codes the tokeniser does not know to X, and truncate."""
    s = "".join(ch for ch in str(seq).upper() if ch.isalpha())
    for bad in "BJOUZ":
        s = s.replace(bad, "X")
    return s[:max_len], len(s) > max_len


class Teacher:
    """Holds the PPLM-Affinity network once and swaps in each fold's weights on demand."""

    def __init__(self, code_dir, weights_path, device_arg="auto"):
        import torch
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from pplm_affinity.model import PPLM_Affinity

        if device_arg == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_arg
        print(f"device: {self.device}")
        t0 = time.time()
        self.cv = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model = PPLM_Affinity(self.cv["pplm_param"])
        self.model.to(self.device)
        self.model.eval()
        self.loaded_fold = None
        self.n_folds = sum(1 for k in self.cv if str(k).startswith("cv"))
        print(f"loaded weights file in {time.time() - t0:.1f}s, folds found: {self.n_folds}")

    def use_fold(self, k):
        if self.loaded_fold != k:
            self.model.load_state_dict(self.cv["cv" + str(k)])
            self.model.to(self.device)
            self.loaded_fold = k

    def predict_pair(self, seq_a, seq_b, folds, symmetrize=True):
        import torch
        vals = []
        for k in folds:
            self.use_fold(k)
            with torch.no_grad():
                v = float(self.model(seq_a, seq_b, self.device).squeeze().cpu().numpy())
                if symmetrize:
                    v = (v + float(self.model(seq_b, seq_a, self.device).squeeze().cpu().numpy())) / 2.0
            vals.append(v)
        return float(np.mean(vals))


def fold_of_pdb(cv_folds_csv):
    """{PDB id (upper) -> fold index} from the authors' published fold list."""
    df = pd.read_csv(cv_folds_csv)
    m = {}
    for k, col in enumerate(df.columns):
        for v in df[col].dropna():
            v = str(v).strip().upper()
            if v:
                m[v] = k
    return m


def run_pairs(args, teacher):
    seqs = read_fasta(os.path.expanduser(args.fasta))
    pairs = pd.read_csv(os.path.expanduser(args.pairs),
                        dtype={"protein1_id": str, "protein2_id": str})
    n_all_pairs = len(pairs)
    all_prot = set(pairs["protein1_id"]) | set(pairs["protein2_id"])
    if args.limit and len(pairs) > args.limit:
        pairs = pairs.sample(n=args.limit, random_state=0).reset_index(drop=True)

    jobs, miss = [], 0
    for _, r in pairs.iterrows():
        a, b = r["protein1_id"], r["protein2_id"]
        if a in seqs and b in seqs:
            jobs.append((a, b))
        else:
            miss += 1
    print(f"pairs to run: {len(jobs):,} ({miss:,} skipped for a missing sequence)")

    folds = list(range(teacher.n_folds))
    rows, t0 = [], time.time()
    for i, (a, b) in enumerate(jobs, 1):
        sa, ta = clean_seq(seqs[a], args.max_len)
        sb, tb = clean_seq(seqs[b], args.max_len)
        v = teacher.predict_pair(sa, sb, folds, symmetrize=not args.no_symmetrize)
        rows.append({"protein1_id": a, "protein2_id": b, "pred_affinity": v,
                     "len1": len(sa), "len2": len(sb), "truncated": int(ta or tb)})
        if i % max(1, args.report_every) == 0 or i == len(jobs):
            el = time.time() - t0
            print(f"  {i:,}/{len(jobs):,} pairs, {el:.1f}s elapsed, {el / i:.2f}s per pair", flush=True)
    dt = time.time() - t0

    res = pd.DataFrame(rows)
    out = os.path.expanduser(args.out) if args.out else os.path.join(
        os.path.dirname(os.path.expanduser(args.pairs)), "candidate_pairs_affinity_pplm.csv")
    res.to_csv(out, index=False)
    print("wrote:", out)

    p = res["pred_affinity"].to_numpy()
    print(f"\npredicted affinity: min {p.min():.3f}, 25% {np.percentile(p, 25):.3f}, "
          f"median {np.median(p):.3f}, 75% {np.percentile(p, 75):.3f}, max {p.max():.3f}")
    print(f"spread: range {p.max() - p.min():.3f}, standard deviation {p.std(ddof=1):.3f}")
    print(f"truncated at {args.max_len} residues: {int(res['truncated'].sum())} of {len(res)} pairs")

    if args.mode == "benchmark":
        per_pair = dt / max(len(jobs), 1)
        wall_h = per_pair * n_all_pairs / 3600.0
        print("\n--- cost extrapolation to the full candidate set ---")
        print(f"  full set: {n_all_pairs:,} pairs over {len(all_prot):,} unique proteins")
        print(f"  measured: {per_pair:.2f}s per pair on this device, "
              f"{teacher.n_folds} fold models x {1 if args.no_symmetrize else 2} chain orders")
        print(f"  full set -> about {wall_h:,.0f} wall-clock hours on one such device "
              f"({wall_h / 24:,.1f} days). Divide by the number of devices run in parallel.")


def run_validate(args, teacher):
    from scipy import stats
    df = pd.read_csv(os.path.expanduser(args.exp_csv))
    df = df[pd.to_numeric(df["pKd"], errors="coerce").notna()].copy()
    df["pKd"] = df["pKd"].astype(float)
    df["complex_pdb"] = df["complex_pdb"].astype(str).str.strip().str.upper()

    folds_map = fold_of_pdb(os.path.join(os.path.expanduser(args.code_dir), "data", "affinity",
                                         "cv_folds.csv"))
    df["teacher_fold"] = df["complex_pdb"].map(folds_map)
    n_in = int(df["teacher_fold"].notna().sum())
    print(f"experimental complexes with a measured pKd: {len(df):,}")
    print(f"  of these, in the teacher's own training folds: {n_in:,} "
          f"({100.0 * n_in / max(len(df), 1):.1f}%), unseen by it: {len(df) - n_in:,}")

    if args.fold_mode == "unseen":
        df = df[df["teacher_fold"].isna()].copy()
        print(f"fold_mode unseen: keeping the {len(df):,} complexes in no fold")
    if args.limit and len(df) > args.limit:
        df = df.sample(n=args.limit, random_state=0).reset_index(drop=True)
    df = df.reset_index(drop=True)

    all_folds = list(range(teacher.n_folds))
    preds, used, t0 = [], [], time.time()
    for i, r in df.iterrows():
        sa, _ = clean_seq(r["seq_receptor"], args.max_len)
        sb, _ = clean_seq(r["seq_ligand"], args.max_len)
        if args.fold_mode == "heldout" and not pd.isna(r["teacher_fold"]):
            folds = [int(r["teacher_fold"])]
        else:
            folds = all_folds
        preds.append(teacher.predict_pair(sa, sb, folds, symmetrize=not args.no_symmetrize))
        used.append(",".join(str(f) for f in folds))
        if (i + 1) % max(1, args.report_every) == 0 or i + 1 == len(df):
            el = time.time() - t0
            print(f"  {i + 1:,}/{len(df):,} complexes, {el:.1f}s elapsed, "
                  f"{el / (i + 1):.2f}s each", flush=True)
    df["pred_affinity"] = preds
    df["folds_used"] = used

    print(f"\nfold_mode: {args.fold_mode}")
    p = df["pred_affinity"].to_numpy()
    print(f"prediction spread: min {p.min():.3f}, median {np.median(p):.3f}, max {p.max():.3f}, "
          f"standard deviation {p.std(ddof=1):.3f}")

    def corr(sub, label):
        m = np.isfinite(sub["pred_affinity"]) & np.isfinite(sub["pKd"])
        if m.sum() < 10:
            print(f"  {label:>22}: n {int(m.sum())}, too few to report")
            return
        pr = stats.pearsonr(sub["pred_affinity"][m], sub["pKd"][m])[0]
        sr = stats.spearmanr(sub["pred_affinity"][m], sub["pKd"][m])[0]
        print(f"  {label:>22}: n {int(m.sum()):>5,}  Pearson {pr:+.3f}  Spearman {sr:+.3f}")

    print("\nprediction vs measured pKd (a useful teacher needs a strong correlation; "
          "the sign depends on the model's output convention)")
    corr(df, "all")
    for s in ["train", "val", "test"]:
        if "split" in df.columns:
            corr(df[df["split"] == s], f"split {s}")
    if args.fold_mode != "unseen" and df["teacher_fold"].isna().any():
        corr(df[df["teacher_fold"].isna()], "unseen by teacher")
        corr(df[df["teacher_fold"].notna()], "in teacher folds")

    out = os.path.expanduser(args.out) if args.out else os.path.join(
        os.path.dirname(os.path.expanduser(args.exp_csv)),
        f"teacher_validation_pplm_{args.fold_mode}.csv")
    keep = [c for c in ["complex_pdb", "split", "pKd", "pred_affinity", "teacher_fold", "folds_used",
                        "len_receptor", "len_ligand"] if c in df.columns]
    df[keep].to_csv(out, index=False)
    print("wrote:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["benchmark", "pairs", "validate"], required=True)
    ap.add_argument("--code_dir", default=DEFAULT_CODE, help="the released PPLM code directory")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="affinity_models.pkl")
    ap.add_argument("--fasta", help="protein FASTA, for benchmark and pairs modes")
    ap.add_argument("--pairs", help="candidate pairs CSV, for benchmark and pairs modes")
    ap.add_argument("--exp_csv", help="experimental_affinity_pairs.csv, for validate mode")
    ap.add_argument("--fold_mode", choices=["heldout", "unseen", "all"], default="heldout")
    ap.add_argument("--limit", type=int, default=None, help="use a random sample of this many rows")
    ap.add_argument("--max_len", type=int, default=1022, help="truncate each sequence to this length")
    ap.add_argument("--no_symmetrize", action="store_true", help="run one chain order only")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--report_every", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 80)
    print(f"Phase C distillation teacher: PPLM-Affinity  (mode: {args.mode})")
    print("=" * 80)

    teacher = Teacher(os.path.expanduser(args.code_dir), os.path.expanduser(args.weights), args.device)
    if args.mode in ("benchmark", "pairs"):
        run_pairs(args, teacher)
    else:
        run_validate(args, teacher)


if __name__ == "__main__":
    main()
