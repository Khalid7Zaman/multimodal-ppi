#!/usr/bin/env python
"""
Phase C distillation, teacher 3: benchmark the released Pythia-PPI model on measured mutation data.

Pythia-PPI predicts how much a single-point mutation CHANGES the binding affinity of a complex (a ddG in
kcal/mol). It is structure-based: it reads a PDB file of the complex and builds a neighbour graph. Two
consequences decide how it can be used here, and both are the point of this benchmark:

  1. It cannot label our STRING candidate pairs. Those pairs have no complex structure, and the model
     predicts a change relative to a wild type rather than an absolute affinity. So it is not a
     replacement for the MM/GBSA route or for an absolute-affinity teacher.
  2. It can be checked honestly on SKEMPI 2.0, which gives measured affinities for the wild type and for
     each mutant of complexes whose structures are known. That is what this script does.

Measured ddG is computed from the two measured dissociation constants in SKEMPI:
      ddG = R T ln(Kd_mutant / Kd_wildtype)
with the temperature taken from the SKEMPI record where it is given, otherwise 298.15 K. A positive value
means the mutation weakens binding.

Caveat to keep in the report: Pythia-PPI was itself trained on SKEMPI, so this measures how well it
reproduces data it has seen, not how it generalises. It is a sanity check on scale, sign and usable
signal, not an independent accuracy figure. SKEMPI complexes are reported per structure, so the honest
reading is the per-complex correlation rather than one pooled number.

Run from /home/zaman/Projects/multimodal-ppi (esmc env):
  python phase_c/teacher_pythia_ppi.py \
      --skempi /home/zaman/Projects/ppi-data/skempi/skempi_v2.csv \
      --pdb_dir /home/zaman/Projects/ppi-data/skempi/PDBs \
      --n_complexes 30 --device cpu
"""
import os
import sys
import time
import math
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CODE = os.path.join(HERE, "teachers", "pythia_ppi", "pythia_code")
R_KCAL = 0.0019872041  # gas constant, kcal per mol per kelvin


def parse_mutation(m):
    """SKEMPI single mutation string, e.g. LI38G: wild-type L, chain I, position 38, mutant G."""
    m = str(m).strip()
    if len(m) < 4:
        return None
    wt, chain, mt, pos = m[0], m[1], m[-1], m[2:-1]
    if not wt.isalpha() or not mt.isalpha() or not chain.isalpha() or not pos:
        return None
    return wt, chain, pos, mt


def load_skempi(path):
    df = pd.read_csv(path, sep=";", low_memory=False)
    need = ["#Pdb", "Mutation(s)_cleaned", "Affinity_mut_parsed", "Affinity_wt_parsed"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"column {c} not found in {path}; columns are: {list(df.columns)[:12]}")
    df = df[df["Mutation(s)_cleaned"].astype(str).str.count(",") == 0].copy()   # single mutations only
    df["kd_mut"] = pd.to_numeric(df["Affinity_mut_parsed"], errors="coerce")
    df["kd_wt"] = pd.to_numeric(df["Affinity_wt_parsed"], errors="coerce")
    df = df[(df["kd_mut"] > 0) & (df["kd_wt"] > 0)].copy()
    temp = df["Temperature"].astype(str).str.extract(r"(\d{3})")[0] if "Temperature" in df.columns else None
    df["T"] = pd.to_numeric(temp, errors="coerce").fillna(298.15) if temp is not None else 298.15
    df.loc[(df["T"] < 250) | (df["T"] > 350), "T"] = 298.15
    df["ddG_measured"] = R_KCAL * df["T"] * np.log(df["kd_mut"] / df["kd_wt"])
    df["pdb_id"] = df["#Pdb"].astype(str).str.split("_").str[0].str.upper()
    parsed = df["Mutation(s)_cleaned"].map(parse_mutation)
    df = df[parsed.notna()].copy()
    df["wt"] = [p[0] for p in parsed[parsed.notna()]]
    df["chain"] = [p[1] for p in parsed[parsed.notna()]]
    df["res_pos"] = [p[2] for p in parsed[parsed.notna()]]
    df["mt"] = [p[3] for p in parsed[parsed.notna()]]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skempi", required=True)
    ap.add_argument("--pdb_dir", required=True, help="directory of SKEMPI PDB files")
    ap.add_argument("--code_dir", default=DEFAULT_CODE)
    ap.add_argument("--n_complexes", type=int, default=30,
                    help="how many distinct complexes to score (sampled, seeded)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 80)
    print("Phase C distillation teacher: Pythia-PPI (mutation ddG, structure-based)")
    print("=" * 80)

    import torch
    from scipy import stats
    code_dir = os.path.expanduser(args.code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from utils.model import Pythia_PPI, get_torch_model
    from utils.dataset import inference_process

    skempi = load_skempi(os.path.expanduser(args.skempi))
    print(f"SKEMPI single-point mutations with both affinities measured: {len(skempi):,} "
          f"over {skempi['pdb_id'].nunique():,} complexes")

    pdb_dir = os.path.expanduser(args.pdb_dir)
    have = {}
    for pid in skempi["pdb_id"].unique():
        for name in (f"{pid}.pdb", f"{pid.lower()}.pdb", f"{pid}.PDB"):
            p = os.path.join(pdb_dir, name)
            if os.path.exists(p):
                have[pid] = p
                break
    print(f"structures found on disk: {len(have):,} of those complexes")
    if not have:
        raise SystemExit(f"no PDB files found in {pdb_dir}")

    rng = np.random.RandomState(0)
    chosen = sorted(have)
    if args.n_complexes and len(chosen) > args.n_complexes:
        chosen = sorted(rng.choice(chosen, size=args.n_complexes, replace=False))
    print(f"scoring {len(chosen)} complexes")

    model = Pythia_PPI(get_torch_model(os.path.join(code_dir, "utils", "pythia", "pythia-p.pt"),
                                       args.device))
    model.load_state_dict(torch.load(os.path.join(code_dir, "train_model", "ppi_affinity.pth"),
                                     map_location=args.device))
    model.to(args.device)
    model.eval()

    rows, t_total, n_struct = [], 0.0, 0
    for pid in chosen:
        try:
            t0 = time.time()
            feats, info = inference_process(have[pid], pid, device=args.device)
            with torch.no_grad():
                pred_aff, _ = model(feats["wt_id"], feats["mt_id"], feats["node_in"], feats["edge_in"])
            dt = time.time() - t0
        except Exception as exc:
            print(f"  {pid}: failed ({type(exc).__name__}: {exc})")
            continue
        t_total += dt
        n_struct += 1
        preds = pred_aff.cpu().numpy().ravel()
        table = {}
        for i in range(len(info["pdb_id"])):
            key = (str(info["chain"][i]), str(info["res_position"][i]).strip(),
                   str(info["wt"][i]), str(info["mt"][i]))
            table[key] = float(preds[i])
        sub = skempi[skempi["pdb_id"] == pid]
        hit = 0
        for _, r in sub.iterrows():
            key = (r["chain"], str(r["res_pos"]).strip(), r["wt"], r["mt"])
            if key in table:
                rows.append({"pdb_id": pid, "chain": r["chain"], "res_pos": r["res_pos"],
                             "wt": r["wt"], "mt": r["mt"],
                             "ddG_measured": float(r["ddG_measured"]),
                             "ddG_pred": table[key]})
                hit += 1
        print(f"  {pid}: {len(table):,} mutations scored in {dt:.1f}s, "
              f"matched {hit} of {len(sub)} measured", flush=True)

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("no measured mutation matched a scored mutation; check chain and numbering")

    print(f"\nscored {n_struct} structures in {t_total:.1f}s "
          f"({t_total / max(n_struct, 1):.1f}s per structure on {args.device})")
    print(f"matched measured mutations: {len(res):,}")

    pr = stats.pearsonr(res["ddG_pred"], res["ddG_measured"])[0]
    sr = stats.spearmanr(res["ddG_pred"], res["ddG_measured"])[0]
    print(f"\npooled over all complexes: n {len(res):,}  Pearson {pr:+.3f}  Spearman {sr:+.3f}")
    print(f"measured ddG: min {res['ddG_measured'].min():+.2f}, median "
          f"{res['ddG_measured'].median():+.2f}, max {res['ddG_measured'].max():+.2f} kcal/mol")
    print(f"predicted:    min {res['ddG_pred'].min():+.2f}, median "
          f"{res['ddG_pred'].median():+.2f}, max {res['ddG_pred'].max():+.2f}")

    per = []
    for pid, sub in res.groupby("pdb_id"):
        if len(sub) >= 10 and sub["ddG_measured"].std() > 0 and sub["ddG_pred"].std() > 0:
            per.append(stats.pearsonr(sub["ddG_pred"], sub["ddG_measured"])[0])
    if per:
        print(f"per-complex Pearson (complexes with at least 10 measured mutations, n={len(per)}): "
              f"median {np.median(per):+.3f}, mean {np.mean(per):+.3f}, "
              f"range {min(per):+.3f} to {max(per):+.3f}")
    else:
        print("per-complex Pearson: no complex had at least 10 matched measured mutations")

    out = os.path.expanduser(args.out) if args.out else os.path.join(
        os.path.dirname(os.path.expanduser(args.skempi)), "teacher_validation_pythia_ppi.csv")
    res.to_csv(out, index=False)
    print("wrote:", out)
    print("\nNOTE: Pythia-PPI was trained on SKEMPI, so this is a memorisation check, not an independent "
          "accuracy figure. It also needs a complex structure, so it cannot label the STRING candidate "
          "pairs at all.")


if __name__ == "__main__":
    main()
