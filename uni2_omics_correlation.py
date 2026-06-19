"""
uni2_omics_correlation.py
=========================
What biology does each UNI2 embedding dimension encode?

For every UNI2 dimension (1,536), correlate its per-patient value against every
molecular feature (immune fractions, immune signatures, RPPA proteins, mutations,
expression, CNV) across the 8-cancer cohort. This is the UNI2 analogue of the
mentor's CTransPath `dim_omic_correlations/` files and uses the SAME output schema
so the two foundation models can be compared directly (Phase 3).

Two correlation modes per omic:
  * pooled       — Spearman across all patients (matches the CTransPath files;
                   optimistic, partly reflects between-cancer differences)
  * residualized — regress each UNI2 dim and each feature on cancer-type dummies
                   first, then correlate the residuals = honest within-cancer signal

Everything runs on CACHED data (no H5 reads):
  * UNI2 features: results_v2/agg_full/full_methods.npz (mean block) + meta.csv
  * omics: the mentor's TCGA multi-omics modules (joined on participant_id)

Thermal-light: each omic loaded once, one vectorized rank-correlation matmul,
cooldown + gc between omics.

Run:
  python uni2_omics_correlation.py --phase 1          # immune, signatures, rppa, mutations
  python uni2_omics_correlation.py --phase 2          # expression, cnv
  python uni2_omics_correlation.py --omics rppa immune # explicit subset
Output: results_v2/dim_omics/uni2_<agg>_<omic>.parquet  (+ summary.csv)
"""
import os
import gc
import argparse
import warnings

import numpy as np
import pandas as pd

from wsi_survival_pipeline import thermal_pause

warnings.filterwarnings("ignore")

NPZ = os.path.join("results_v2", "agg_full", "full_methods.npz")
META = os.path.join("results_v2", "agg_full", "meta.csv")
OMICS_DIR_DEFAULT = os.path.join("..", "TCGA multiomics", "TCGA_MODULES_FULL")
OUT_DIR = os.path.join("results_v2", "dim_omics")
SUMMARY = os.path.join(OUT_DIR, "summary.csv")

# per-omic config: (parquet file, feature-column prefix, special handling)
OMIC_SPECS = {
    "immune":            ("IMMUNE_DECONVOLUTION.parquet", "immune_", None),
    "immune_signatures": ("IMMUNE_SIGNATURES.parquet",    "sig_",    None),
    "rppa":              ("RPPA_FULL.parquet",            "protein.", "ragged"),
    "mutations":         ("MUTATIONS_FULL.parquet",       "bin_",    "mutation"),
    "expression":        ("EXPRESSION_FULL_WITH_SYMBOLS.parquet", "expr_", "variable"),
    "cnv":               ("CNV_FULL.parquet",             "cnv_",    "variable"),
}
PHASES = {1: ["immune", "immune_signatures", "rppa", "mutations"],
          2: ["expression", "cnv"]}

MIN_MUT_PATIENTS = 30     # drop genes mutated in fewer than this many cohort patients
TOP_VARIABLE = 4000       # expression/CNV: keep this many most-variable features


# ----------------------------------------------------------------- helpers
def bh_qvalues(p):
    """Benjamini-Hochberg FDR q-values for a flat array of p-values."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def rank_z(M):
    """Column-wise rank-transform (average ties) then z-score. M: (n x k)."""
    R = pd.DataFrame(M).rank(axis=0).to_numpy()
    R -= R.mean(0)
    sd = R.std(0)
    sd[sd == 0] = 1.0
    return R / sd


def residualize(M, D):
    """Remove cancer-type means: residual of M regressed on dummy design D."""
    beta, *_ = np.linalg.lstsq(D, M, rcond=None)
    return M - D @ beta


def spearman_matrix(X, Y, n):
    """rho (kx_dims x ky_feats) + two-sided p from rank-z matrices X, Y."""
    from scipy.stats import t as tdist
    rho = (X.T @ Y) / n
    rho = np.clip(rho, -0.999999, 0.999999)
    tstat = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    p = 2 * tdist.sf(np.abs(tstat), n - 2)
    return rho, p


# ----------------------------------------------------------------- data
def load_uni2_patient_matrix(agg):
    d = np.load(NPZ, allow_pickle=True)
    methods = [str(m) for m in d["method_names"]]
    if agg not in methods:
        raise ValueError(f"aggregation '{agg}' not in npz ({methods})")
    bd = int(d["block_dim"])
    block = d["features"][:, methods.index(agg) * bd:(methods.index(agg) + 1) * bd]
    meta = pd.read_csv(META)
    tum = meta["sample_type"].values == "tumor"
    block, pids = block[tum], meta["participant_id"].values[tum]
    # collapse slides -> one vector per patient (mean over their slides)
    uniq = pd.unique(pids)
    X = np.vstack([block[pids == p].mean(0) for p in uniq])
    cols = [f"dim_{i}" for i in range(bd)]
    print(f"UNI2 '{agg}': {X.shape[0]} patients x {bd} dims", flush=True)
    return pd.DataFrame(X, index=uniq, columns=cols)


def load_omic(name, omics_dir):
    """Return the omic's numeric feature columns (by prefix), indexed by participant."""
    fn, prefix, _ = OMIC_SPECS[name]
    df = pd.read_parquet(os.path.join(omics_dir, fn))
    if df.index.name != "participant_id" and "participant_id" in df.columns:
        df = df.set_index("participant_id")
    df = df[~df.index.duplicated(keep="first")]   # some modules have >1 sample/patient
    feat_cols = [c for c in df.columns if c.startswith(prefix)]
    Y = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    if name == "rppa":
        mp = os.path.join(omics_dir, "ARCHIVE", "RPPA_numeric_column_mapping.csv")
        if os.path.exists(mp):
            m = pd.read_csv(mp)
            ren = {r["rppa_column"]: "rppa_" + str(r["protein_name"]).lstrip("X")
                   for _, r in m.iterrows()}
            Y = Y.rename(columns=ren)
            Y = Y.loc[:, ~Y.columns.duplicated()]   # mapping isn't 1:1 -> keep first
    return Y


def select_features(Yc, special, n):
    """Cohort-based feature filtering applied AFTER patient alignment."""
    if special == "mutation":
        thr = max(MIN_MUT_PATIENTS, int(0.05 * n))
        freq = (Yc > 0).sum(axis=0)
        keep = freq[freq >= thr].sort_values(ascending=False).head(400).index
        Yc = Yc[keep]
        print(f"  mutations: kept {len(keep)} genes mutated in >= {thr} cohort patients "
              f"(top 400 by frequency)", flush=True)
    elif special == "variable":
        keep = Yc.var(axis=0).sort_values(ascending=False).head(TOP_VARIABLE).index
        Yc = Yc[keep]
        print(f"  kept top {Yc.shape[1]} most-variable features", flush=True)
    return Yc


# ----------------------------------------------------------------- core
def _summ_row(name, mode, df):
    sig = int((df["q_value"] < 0.05).sum())
    top = df.loc[df["rho"].abs().idxmax()]
    print(f"    {mode:12s}: {sig} pairs q<0.05 | strongest {top['dim']} ~ "
          f"{top['feature']} rho={top['rho']:.3f} (n={int(top['n'])})", flush=True)
    return {"omic": name, "mode": mode, "n_patients": int(df["n"].max()),
            "n_features": df["feature"].nunique(), "n_pairs": len(df),
            "n_sig_q05": sig, "top_dim": top["dim"], "top_feature": top["feature"],
            "top_rho": round(float(top["rho"]), 3)}


def _design(cancer_idx_values, purity, idx):
    """Cancer-dummy design, optionally augmented with a z-scored purity column."""
    D = pd.get_dummies(pd.Series(cancer_idx_values)).to_numpy(float)
    if purity is not None:
        pv = purity.reindex(idx).to_numpy(float)
        mu = np.nanmean(pv) if np.isfinite(np.nanmean(pv)) else 0.0
        pv = np.where(np.isnan(pv), mu, pv)
        sd = pv.std() or 1.0
        D = np.column_stack([D, (pv - pv.mean()) / sd])
    return D


def correlate_ragged(name, Xc, Yc, cancer, agg, min_n=50, purity=None):
    """Per-feature correlation for sparsely-measured omics (RPPA): each feature
    uses only the patients where it was actually measured (no imputation)."""
    Yc = Yc.loc[:, Yc.notna().sum(axis=0) >= min_n]
    dims = list(Xc.columns)
    print(f"  {name} (ragged): {Yc.shape[1]} features measured in >= {min_n} patients",
          flush=True)
    modes = ["pooled", "residualized"] + (["residualized_purity"] if purity is not None else [])
    rows, summ = [], []
    for mode in modes:
        recs = []
        for feat in Yc.columns:
            yv = Yc[feat].dropna()
            n = len(yv)
            if n < min_n:
                continue
            Xs = Xc.loc[yv.index].to_numpy(float)
            yy = yv.to_numpy(float).reshape(-1, 1)
            if mode in ("residualized", "residualized_purity"):
                D = _design(cancer.loc[yv.index].values,
                            purity if mode == "residualized_purity" else None, yv.index)
                Xs, yy = residualize(Xs, D), residualize(yy, D)
            rho, p = spearman_matrix(rank_z(Xs), rank_z(yy), n)
            recs.append(pd.DataFrame({"dim": dims, "feature": feat,
                                      "rho": rho[:, 0], "p_value": p[:, 0], "n": n}))
        df = pd.concat(recs, ignore_index=True)
        df["q_value"] = bh_qvalues(df["p_value"].values)
        df["model"], df["aggregation"], df["mode"] = "uni2", agg, mode
        df = df[["dim", "feature", "rho", "p_value", "q_value", "n",
                 "model", "aggregation", "mode"]]
        rows.append(df)
        summ.append(_summ_row(name, mode, df))
    return rows, summ


def correlate_omic(name, Xpat, cancer, omics_dir, agg, purity=None):
    special = OMIC_SPECS[name][2]
    Y = load_omic(name, omics_dir)
    common = Xpat.index.intersection(Y.index)
    Xc, Yc = Xpat.loc[common], Y.loc[common]
    if special == "ragged":
        return correlate_ragged(name, Xc, Yc, cancer.loc[common], agg, purity=purity)
    # NaN-robust: drop all-empty features, keep patients with >=50% features measured,
    # then median-impute remaining gaps (handles RPPA's sparse protein coverage)
    Yc = Yc.loc[:, Yc.notna().any(axis=0)]
    keep_pat = (Yc.notna().mean(axis=1) >= 0.5).values
    Xc, Yc = Xc[keep_pat], Yc[keep_pat]
    Yc = Yc.fillna(Yc.median(axis=0))
    n = len(Xc)
    if n < 30:
        print(f"  {name}: only {n} usable patients — skipping", flush=True)
        return [], []
    Yc = select_features(Yc, special, n)
    canc = cancer.loc[Xc.index]
    Xv, Yv = Xc.to_numpy(float), Yc.to_numpy(float)
    dims, feats = list(Xc.columns), list(Yc.columns)
    D = pd.get_dummies(canc.values).to_numpy(float)
    print(f"  {name}: {n} patients x {len(feats)} features ({len(np.unique(canc))} cancers)",
          flush=True)

    modes = [("pooled", Xv, Yv),
             ("residualized", residualize(Xv, D), residualize(Yv, D))]
    if purity is not None:
        Dp = _design(canc.values, purity, Xc.index)
        modes.append(("residualized_purity", residualize(Xv, Dp), residualize(Yv, Dp)))
    rows, summ = [], []
    for mode, Xm, Ym in modes:
        rho, p = spearman_matrix(rank_z(Xm), rank_z(Ym), n)
        q = bh_qvalues(p.ravel()).reshape(rho.shape)
        df = pd.DataFrame({
            "dim": np.repeat(dims, len(feats)),
            "feature": np.tile(feats, len(dims)),
            "rho": rho.ravel(), "p_value": p.ravel(), "q_value": q.ravel(),
            "n": n, "model": "uni2", "aggregation": agg, "mode": mode,
        })
        rows.append(df)
        sig = int((df["q_value"] < 0.05).sum())
        top = df.loc[df["rho"].abs().idxmax()]
        summ.append({"omic": name, "mode": mode, "n_patients": n,
                     "n_features": len(feats), "n_pairs": len(df),
                     "n_sig_q05": sig, "top_dim": top["dim"],
                     "top_feature": top["feature"], "top_rho": round(float(top["rho"]), 3)})
        print(f"    {mode:12s}: {sig} pairs q<0.05 | strongest {top['dim']} ~ "
              f"{top['feature']} rho={top['rho']:.3f}", flush=True)
    del Y, Yc, Yv
    gc.collect()
    return rows, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2], default=1)
    ap.add_argument("--omics", nargs="*", default=None,
                    help="explicit omic names (overrides --phase)")
    ap.add_argument("--agg", default="mean")
    ap.add_argument("--omics-dir", default=OMICS_DIR_DEFAULT)
    ap.add_argument("--cooldown", type=float, default=1.5)
    ap.add_argument("--purity", action="store_true",
                    help="also emit a residualized_purity mode (control for tumor purity)")
    ap.add_argument("--purity-col", default="purity_consensus")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    omics = args.omics if args.omics else PHASES[args.phase]
    print(f"Omics: {omics} | aggregation: {args.agg}", flush=True)

    Xpat = load_uni2_patient_matrix(args.agg)
    clin = pd.read_parquet(os.path.join(args.omics_dir, "CLINICAL_ENHANCED.parquet"))
    if clin.index.name != "participant_id" and "participant_id" in clin.columns:
        clin = clin.set_index("participant_id")
    clin = clin[~clin.index.duplicated(keep="first")]
    cancer = clin["project_id"].reindex(Xpat.index)
    Xpat = Xpat.loc[cancer.notna().values]           # keep patients with a cancer label
    cancer = cancer.loc[Xpat.index]

    purity = None
    if args.purity:
        pc = pd.read_parquet(os.path.join(args.omics_dir, "PURITY_CONSOLIDATED.parquet"))
        if pc.index.name != "participant_id" and "participant_id" in pc.columns:
            pc = pc.set_index("participant_id")
        pc = pc[~pc.index.duplicated(keep="first")]
        col = args.purity_col if args.purity_col in pc.columns else "purity_cpe"
        purity = pd.to_numeric(pc[col], errors="coerce")
        print(f"purity covariate: {col} ({int(purity.notna().sum())} patients)", flush=True)

    all_summ = []
    for name in omics:
        print(f"\n=== {name} ===", flush=True)
        rows, summ = correlate_omic(name, Xpat, cancer, args.omics_dir, args.agg, purity=purity)
        if rows:
            out = os.path.join(OUT_DIR, f"uni2_{args.agg}_{name}.parquet")
            pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
            print(f"  wrote {out}", flush=True)
            all_summ.extend(summ)
        gc.collect()
        thermal_pause(args.cooldown)

    if all_summ:
        s = pd.DataFrame(all_summ)
        if os.path.exists(SUMMARY):
            prev = pd.read_csv(SUMMARY)
            prev = prev[~prev.set_index(["omic", "mode"]).index.isin(
                s.set_index(["omic", "mode"]).index)]
            s = pd.concat([prev, s], ignore_index=True)
        s.to_csv(SUMMARY, index=False)
        print(f"\nwrote {SUMMARY}\n{s.to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
