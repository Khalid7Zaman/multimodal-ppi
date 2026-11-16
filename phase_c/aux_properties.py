#!/usr/bin/env python
"""
Phase C: additional experimental properties that correlate with binding affinity, with numbers.

This is the third of the three analyses the supervisor asked for before scaling up synthetic generation.
Point three is specifically about other MEASURED (experimental) properties, not computed values, that move
together with binding affinity. If such properties exist and correlate with affinity, they can be added as
auxiliary prediction tasks, which usually improves the main affinity prediction and reuses data that
already exists.

The evidence comes from SKEMPI 2.0, a public benchmark of measured protein-protein binding energy,
kinetics and thermodynamics. For the same complexes it reports, from real experiments: the dissociation
constant (affinity), the association and dissociation rate constants kon and koff, and the calorimetric
enthalpy and entropy of binding. This script parses that file, turns affinity into a binding free energy,
and measures how strongly each other property correlates with affinity, both across complexes (wild type)
and across single mutations (the change in each property upon mutation). It writes a correlation table and
a figure. CPU only, seconds to run.

Input file: skempi_v2.csv, downloaded from
    https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv

Run:
    python phase_c/aux_properties.py --csv <path to skempi_v2.csv> --out <output dir>
writes skempi_property_correlations.csv and aux_properties.png in --out.
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

R_KCAL = 8.314 / 4184.0  # gas constant in kcal / mol / K, as the SKEMPI FAQ defines it
NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def to_float(x):
    """Pull the first numeric token out of a messy cell, else NaN. Handles units, flags, blanks."""
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ("na", "nan", "n.b.", "n.d.", "none"):
        return np.nan
    m = NUM.search(s.replace(",", ""))
    return float(m.group()) if m else np.nan


def find_col(cols, *needles):
    """First column whose name contains all needles (case-insensitive)."""
    low = {c: c.lower() for c in cols}
    for c in cols:
        if all(n.lower() in low[c] for n in needles):
            return c
    return None


def dG(kd, T):
    """Binding free energy in kcal/mol from Kd (molar) and temperature (K). More negative = stronger."""
    kd = np.asarray(kd, dtype=float)
    out = np.full(kd.shape, np.nan)
    ok = np.isfinite(kd) & (kd > 0)
    out[ok] = R_KCAL * T[ok] * np.log(kd[ok])
    return out


def corr_row(name, x, y):
    """Pearson and Spearman of two aligned arrays, on the rows where both are finite."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 10:
        return dict(property=name, n=n, pearson_r=np.nan, pearson_p=np.nan,
                    spearman_r=np.nan, spearman_p=np.nan)
    pr = stats.pearsonr(x[m], y[m])
    sr = stats.spearmanr(x[m], y[m])
    return dict(property=name, n=n, pearson_r=pr[0], pearson_p=pr[1],
                spearman_r=sr[0], spearman_p=sr[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to skempi_v2.csv")
    ap.add_argument("--out", default=None, help="output directory (default: the csv directory)")
    args = ap.parse_args()
    out = args.out or (os.path.dirname(os.path.abspath(args.csv)) or ".")

    # delimiter auto-detect from the header line
    with open(args.csv, "r", encoding="utf-8", errors="replace") as f:
        head = f.readline()
    sep = ";" if head.count(";") >= head.count(",") else ","
    df = pd.read_csv(args.csv, sep=sep, dtype=str, encoding="utf-8",
                     engine="python", on_bad_lines="skip")
    cols = list(df.columns)
    print("=" * 80)
    print("Phase C: experimental properties correlating with affinity (SKEMPI 2.0)")
    print("=" * 80)
    print(f"rows: {len(df):,}   delimiter: '{sep}'   columns: {len(cols)}")

    # locate columns (robust to the exact header spelling and to optional parsed columns)
    c_aff_wt = find_col(cols, "affinity_wt") or find_col(cols, "affinity", "wt")
    c_aff_mt = find_col(cols, "affinity_mut") or find_col(cols, "affinity", "mut")
    c_kon_wt = find_col(cols, "kon_wt") or find_col(cols, "kon", "wt")
    c_kon_mt = find_col(cols, "kon_mut") or find_col(cols, "kon", "mut")
    c_koff_wt = find_col(cols, "koff_wt") or find_col(cols, "koff", "wt")
    c_koff_mt = find_col(cols, "koff_mut") or find_col(cols, "koff", "mut")
    c_dh_wt = find_col(cols, "dh_wt") or find_col(cols, "dh", "wt")
    c_ds_wt = find_col(cols, "ds_wt") or find_col(cols, "ds", "wt")
    c_temp = find_col(cols, "temperature") or find_col(cols, "temp")
    print("matched columns:")
    for lab, c in [("affinity_wt", c_aff_wt), ("affinity_mut", c_aff_mt), ("kon_wt", c_kon_wt),
                   ("koff_wt", c_koff_wt), ("dH_wt", c_dh_wt), ("dS_wt", c_ds_wt),
                   ("Temperature", c_temp)]:
        print(f"  {lab:>13} -> {c}")

    # parse numeric values
    T = df[c_temp].map(to_float).to_numpy() if c_temp else np.full(len(df), np.nan)
    T = np.where(np.isfinite(T) & (T > 150) & (T < 400), T, 298.15)  # default room temperature
    kd_wt = df[c_aff_wt].map(to_float).to_numpy()
    kd_mt = df[c_aff_mt].map(to_float).to_numpy() if c_aff_mt else np.full(len(df), np.nan)
    kon_wt = df[c_kon_wt].map(to_float).to_numpy() if c_kon_wt else np.full(len(df), np.nan)
    koff_wt = df[c_koff_wt].map(to_float).to_numpy() if c_koff_wt else np.full(len(df), np.nan)
    koff_mt = df[c_koff_mt].map(to_float).to_numpy() if c_koff_mt else np.full(len(df), np.nan)
    kon_mt = df[c_kon_mt].map(to_float).to_numpy() if c_kon_mt else np.full(len(df), np.nan)
    dh_wt = df[c_dh_wt].map(to_float).to_numpy() if c_dh_wt else np.full(len(df), np.nan)
    ds_wt = df[c_ds_wt].map(to_float).to_numpy() if c_ds_wt else np.full(len(df), np.nan)

    dg_wt = dG(kd_wt, T)
    dg_mt = dG(kd_mt, T)
    log_koff_wt = np.log10(np.where(koff_wt > 0, koff_wt, np.nan))
    log_kon_wt = np.log10(np.where(kon_wt > 0, kon_wt, np.nan))
    tds_wt = (T * ds_wt) / 1000.0  # cal/mol/K times K -> cal/mol -> kcal/mol
    dg_from_hs = dh_wt - tds_wt     # thermodynamic identity check

    print(f"\nproperty availability (entries with a usable wild-type value):")
    for lab, v in [("affinity (Kd)", kd_wt), ("kon", kon_wt), ("koff", koff_wt),
                   ("dH", dh_wt), ("dS", ds_wt)]:
        print(f"  {lab:>14}: {int(np.isfinite(v).sum()):>6,}")

    # ---- correlations of each measured property with affinity (wild type, across complexes) ----
    rows = []
    rows.append(corr_row("log10(koff) vs dG", log_koff_wt, dg_wt))
    rows.append(corr_row("log10(kon) vs dG", log_kon_wt, dg_wt))
    rows.append(corr_row("dH vs dG", dh_wt, dg_wt))
    rows.append(corr_row("T*dS vs dG", tds_wt, dg_wt))
    rows.append(corr_row("(dH - T*dS) vs dG", dg_from_hs, dg_wt))

    # ---- mutation effect: does the CHANGE in each property track the change in affinity ----
    ddg = dg_mt - dg_wt
    dlog_koff = np.log10(np.where(koff_mt > 0, koff_mt, np.nan)) - log_koff_wt
    dlog_kon = np.log10(np.where(kon_mt > 0, kon_mt, np.nan)) - log_kon_wt
    rows.append(corr_row("d log10(koff) vs ddG", dlog_koff, ddg))
    rows.append(corr_row("d log10(kon) vs ddG", dlog_kon, ddg))

    tab = pd.DataFrame(rows)
    tab_path = os.path.join(out, "skempi_property_correlations.csv")
    tab.to_csv(tab_path, index=False)

    print("\nCorrelation with binding affinity (dG; mutation rows use ddG):")
    print(f"  {'property':<24} {'n':>7} {'pearson_r':>10} {'spearman_r':>11}")
    for _, r in tab.iterrows():
        print(f"  {r['property']:<24} {int(r['n']):>7,} {r['pearson_r']:>10.3f} {r['spearman_r']:>11.3f}")

    # ---- figure: four scatter panels, each annotated with the correlation ----
    def panel(ax, x, y, xlabel, ylabel, title):
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=7, alpha=0.3, c="#4c78a8")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        if m.sum() >= 10:
            r = stats.pearsonr(x[m], y[m])[0]
            ax.text(0.04, 0.93, f"r = {r:.2f}  n = {int(m.sum()):,}",
                    transform=ax.transAxes, fontsize=9, va="top")

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    panel(ax[0], log_koff_wt, dg_wt, "log10 koff (1/s)", "binding dG (kcal/mol)",
          "Dissociation rate vs affinity")
    panel(ax[1], log_kon_wt, dg_wt, "log10 kon (1/M/s)", "binding dG (kcal/mol)",
          "Association rate vs affinity")
    panel(ax[2], dg_from_hs, dg_wt, "dH - T*dS (kcal/mol)", "binding dG (kcal/mol)",
          "Thermodynamics vs affinity")
    panel(ax[3], dlog_koff, ddg, "d log10 koff upon mutation", "ddG upon mutation (kcal/mol)",
          "Mutation: koff change vs affinity change")
    fig.tight_layout()
    figp = os.path.join(out, "aux_properties.png")
    fig.savefig(figp, dpi=130)

    print("\nWrote:")
    print("  ", tab_path)
    print("  ", figp)


if __name__ == "__main__":
    main()
