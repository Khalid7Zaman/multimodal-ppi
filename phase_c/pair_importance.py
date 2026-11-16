#!/usr/bin/env python
"""
Phase C: PPI-pair importance score for the de-leaked candidate set, with numerical evidence.

This is the second of the three analyses the supervisor asked for before scaling up synthetic
generation. It ranks every candidate interacting pair by how valuable it is to generate a synthetic
label for, so the generation budget can be spent on the pairs that matter most instead of a random
sample. The score combines three transparent, standard axes, each turned into a percentile across all
candidate pairs so no axis dominates by its raw scale:

  1. Reliability. The STRING combined physical score of the edge (700 to 1000). A higher score means the
     interaction is better supported, so a synthetic label on it is more trustworthy.
  2. Centrality. How central the two proteins are in the candidate interaction network, measured by
     degree and by weighted PageRank. Central proteins are the hubs of the interactome, which are
     enriched for essential and disease-related functions (the centrality and lethality relationship),
     so modelling their interactions well has more downstream impact.
  3. Novelty. How far the two proteins sit from the experimental proteins in the model's own ESM-C
     representation, in the same PCA frame used for the coverage analysis. Novel pairs are the ones that
     extend the sequence space beyond the experimental envelope, which is the supervisor's stated goal.

The three axes are averaged with equal weight by default. Each axis is written out separately so the
weighting can be changed and defended. The script writes the full scored table and a three-panel figure,
and prints the headline numbers. CPU only, no GPU needed, because it reuses the embeddings already
computed for the coverage analysis.

Run from ~/Projects/multimodal-ppi (esmc env):
    python phase_c/pair_importance.py \
        --pairs ~/Projects/ppi-data/phase_c/candidate_pairs_clean.csv \
        --exp   ~/Projects/ppi-data/phase_c/emb_experimental \
        --syn   ~/Projects/ppi-data/phase_c/emb_synthetic \
        --out   ~/Projects/ppi-data/phase_c
writes candidate_pairs_scored.csv and pair_importance.png in --out.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def load_embeddings(prefix):
    """Load <prefix>.npy and <prefix>.ids.txt, return (matrix, id list)."""
    X = np.load(prefix + ".npy")
    with open(prefix + ".ids.txt") as f:
        ids = [line.strip() for line in f if line.strip()]
    if len(ids) != len(X):
        raise SystemExit(f"{prefix}: {len(ids)} ids but {len(X)} vectors, they must match")
    return X, ids


def pagerank(n, src, dst, weight, damping=0.85, iters=200, tol=1e-10):
    """Weighted PageRank on an undirected graph, by sparse power iteration. No networkx needed."""
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    w = np.concatenate([weight, weight]).astype(np.float64)
    A = sparse.csr_matrix((w, (rows, cols)), shape=(n, n))
    outstrength = np.asarray(A.sum(axis=1)).ravel()
    dangling = outstrength == 0
    inv = np.zeros(n)
    inv[~dangling] = 1.0 / outstrength[~dangling]
    teleport = np.ones(n) / n
    r = np.ones(n) / n
    for _ in range(iters):
        prev = r
        contrib = inv * r
        r = damping * (A @ contrib) + damping * r[dangling].sum() * teleport + (1 - damping) * teleport
        r = r / r.sum()
        if np.abs(r - prev).sum() < tol:
            break
    return r


def rank01(x):
    """Percentile rank of each value in [0, 1] (average rank, ties shared)."""
    s = pd.Series(x)
    return (s.rank(method="average") - 1) / max(len(s) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="candidate_pairs_clean.csv")
    ap.add_argument("--exp", required=True, help="prefix of experimental embeddings (<prefix>.npy)")
    ap.add_argument("--syn", required=True, help="prefix of synthetic (candidate) embeddings")
    ap.add_argument("--out", default=None, help="output directory (default: the --pairs directory)")
    ap.add_argument("--k", type=int, default=10, help="PCA dimensions for the novelty space")
    ap.add_argument("--w_reliability", type=float, default=1.0)
    ap.add_argument("--w_centrality", type=float, default=1.0)
    ap.add_argument("--w_novelty", type=float, default=1.0)
    args = ap.parse_args()

    out = args.out or (os.path.dirname(args.pairs) or ".")
    df = pd.read_csv(args.pairs, dtype={"protein1_id": str, "protein2_id": str,
                                        "protein1": str, "protein2": str, "taxid": str})
    print("=" * 78)
    print("Phase C: PPI-pair importance score (reliability, centrality, novelty)")
    print("=" * 78)
    print(f"candidate pairs: {len(df):,}   columns: {list(df.columns)}")

    # ---- node index over the proteins that appear in the candidate pairs ----
    nodes = pd.Index(pd.unique(pd.concat([df["protein1_id"], df["protein2_id"]], ignore_index=True)))
    node_pos = {p: i for i, p in enumerate(nodes)}
    n = len(nodes)
    src = df["protein1_id"].map(node_pos).to_numpy()
    dst = df["protein2_id"].map(node_pos).to_numpy()
    score = df["combined_score"].to_numpy(dtype=np.float64)
    print(f"candidate proteins (network nodes): {n:,}   edges: {len(df):,}")

    # ---- centrality: unweighted degree and score-weighted PageRank ----
    deg = np.zeros(n, dtype=np.int64)
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)
    pr = pagerank(n, src, dst, score / 1000.0)
    print(f"degree: max {deg.max():,}, median {int(np.median(deg))}   "
          f"PageRank: max {pr.max():.2e}, median {np.median(pr):.2e}")

    # ---- novelty: distance to the nearest experimental protein in the ESM-C / PCA frame ----
    Xe, _ = load_embeddings(args.exp)
    Xs, syn_ids = load_embeddings(args.syn)
    both = np.vstack([Xe, Xs])
    sc = StandardScaler().fit(both)
    pca = PCA(n_components=args.k).fit(sc.transform(both))
    Pe = pca.transform(sc.transform(Xe))
    Ps = pca.transform(sc.transform(Xs))

    # typical experimental spacing, so novelty is in units of normal protein-to-protein distance
    nne = NearestNeighbors(n_neighbors=2).fit(Pe)
    d_ee, _ = nne.kneighbors(Pe)
    exp_spacing = float(np.median(d_ee[:, 1]))
    nn = NearestNeighbors(n_neighbors=1).fit(Pe)
    d_se, _ = nn.kneighbors(Ps)
    novelty_syn = (d_se[:, 0] / max(exp_spacing, 1e-9))
    nov_of_id = dict(zip(syn_ids, novelty_syn))

    missing = [p for p in nodes if p not in nov_of_id]
    if missing:
        print(f"WARNING: {len(missing):,} candidate proteins have no embedding, "
              f"their novelty is set to the median")
    med_nov = float(np.median(novelty_syn))
    node_novelty = np.array([nov_of_id.get(p, med_nov) for p in nodes])
    print(f"novelty (protein to nearest experimental, in units of experimental spacing): "
          f"median {np.median(node_novelty):.2f}, 95th pct {np.percentile(node_novelty, 95):.2f}")

    # ---- per-pair components ----
    df["confidence"] = score / 1000.0
    df["deg1"] = deg[src]
    df["deg2"] = deg[dst]
    df["pr1"] = pr[src]
    df["pr2"] = pr[dst]
    df["nov1"] = node_novelty[src]
    df["nov2"] = node_novelty[dst]
    df["pair_pagerank"] = df["pr1"] + df["pr2"]
    df["pair_degree"] = df["deg1"] + df["deg2"]
    df["pair_novelty"] = (df["nov1"] + df["nov2"]) / 2.0

    # ---- three axes, each as a percentile across all pairs, then an equal-weight blend ----
    df["reliability_rank"] = rank01(df["confidence"])
    df["centrality_rank"] = rank01(df["pair_pagerank"])
    df["novelty_rank"] = rank01(df["pair_novelty"])
    wsum = args.w_reliability + args.w_centrality + args.w_novelty
    df["importance"] = (args.w_reliability * df["reliability_rank"]
                        + args.w_centrality * df["centrality_rank"]
                        + args.w_novelty * df["novelty_rank"]) / wsum
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    cols = ["rank", "pair_key", "taxid", "protein1_id", "protein2_id", "combined_score", "confidence",
            "pair_degree", "pair_pagerank", "pair_novelty",
            "reliability_rank", "centrality_rank", "novelty_rank", "importance"]
    cols = [c for c in cols if c in df.columns]
    scored_path = os.path.join(out, "candidate_pairs_scored.csv")
    df[cols].to_csv(scored_path, index=False)

    # ---- headline numbers: does the top selection really pick more novel, more central pairs ----
    print()
    print("Top selection versus a random selection of the same size:")
    print(f"  {'N':>8} {'top novelty':>12} {'rand novelty':>13} {'top pagerank':>14} {'rand pagerank':>15}")
    rng = np.random.default_rng(0)
    sizes = [s for s in [5000, 10000, 20000, 40000, 80000] if s < len(df)]
    nov_top, nov_rand = [], []
    for s in sizes:
        top = df.iloc[:s]
        ridx = rng.choice(len(df), size=s, replace=False)
        rnd = df.iloc[ridx]
        nov_top.append(top["pair_novelty"].mean())
        nov_rand.append(rnd["pair_novelty"].mean())
        print(f"  {s:>8,} {top['pair_novelty'].mean():>12.3f} {rnd['pair_novelty'].mean():>13.3f} "
              f"{top['pair_pagerank'].mean():>14.2e} {rnd['pair_pagerank'].mean():>15.2e}")

    print()
    print("Score composition of the overall top 1000 pairs (mean percentile on each axis):")
    t1k = df.iloc[:1000]
    print(f"  reliability {t1k['reliability_rank'].mean():.2f}   "
          f"centrality {t1k['centrality_rank'].mean():.2f}   "
          f"novelty {t1k['novelty_rank'].mean():.2f}")
    print()

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    ax[0].hist(df["importance"], bins=60, color="#4c78a8")
    ax[0].set_xlabel("importance score (0 to 1)")
    ax[0].set_ylabel("candidate pairs")
    ax[0].set_title("Distribution of pair importance")

    samp = df.sample(n=min(20000, len(df)), random_state=0)
    sctr = ax[1].scatter(samp["pair_novelty"], samp["pair_pagerank"], c=samp["importance"],
                         s=6, alpha=0.4, cmap="viridis")
    ax[1].set_xlabel("pair novelty (experimental spacings)")
    ax[1].set_ylabel("pair PageRank (centrality)")
    ax[1].set_yscale("log")
    ax[1].set_title("Importance across novelty and centrality")
    cb = fig.colorbar(sctr, ax=ax[1]); cb.set_label("importance")

    xs = np.array(sizes)
    ax[2].plot(xs, nov_top, "o-", color="#e45756", label="top by importance")
    ax[2].plot(xs, nov_rand, "s--", color="#4c78a8", label="random selection")
    ax[2].set_xlabel("pairs selected (N)")
    ax[2].set_ylabel("mean novelty of selection")
    ax[2].set_title("Ranked selection extends the sequence space")
    ax[2].legend(fontsize=9)

    fig.tight_layout()
    figp = os.path.join(out, "pair_importance.png")
    fig.savefig(figp, dpi=130)
    print("Wrote:")
    print("  ", scored_path)
    print("  ", figp)


if __name__ == "__main__":
    main()
