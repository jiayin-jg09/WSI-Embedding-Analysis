"""
uni2_predictability.py
======================
How much of the tumor's molecular state can the WHOLE UNI2 embedding predict?

The dim-omics correlation work was univariate (one dimension vs one feature). This
asks the multivariate question: train a cross-validated model on all 1,536 UNI2
dimensions and measure how well it predicts each molecular target. Two embedding
blocks are compared:
  * mean  — the usual slide vector (Upgrade A: general predictability)
  * std   — intra-slide embedding heterogeneity (Upgrade B: does histological
            heterogeneity predict genomic instability?)

Targets (per patient, from the lab's TCGA modules):
  immune signatures, immune cell fractions, tumor-purity metrics, genomic-
  instability scores (TMB, aneuploidy, subclonal entropy, ...), and a few driver
  mutations (binary).

CV = cancer-stratified 5-fold. For each target we report BOTH:
  * pooled metric        — across all patients (cancer-type confounded)
  * within-cancer metric — per-cancer means removed (R2) / per-cancer AUC averaged
The within-cancer number is the honest one (same idea as the stratified C-index).

Thermal-light: PCA fit once per fold (shared across targets), Ridge is closed-form.
No H5 reads. Run: python uni2_predictability.py
Output: results_v2/dim_omics/predictability.csv
"""
import os
import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score

from wsi_survival_pipeline import thermal_pause
from uni2_omics_correlation import load_uni2_patient_matrix, OMICS_DIR_DEFAULT

warnings.filterwarnings("ignore")

OUT = os.path.join("results_v2", "dim_omics", "predictability.csv")
BLOCKS = ["mean", "std"]
PURITY_COLS = ["purity_consensus", "purity_cpe", "purity_absolute", "purity_ploidy",
               "purity_aneuploidy_score", "purity_loh_frac_altered"]
INSTABILITY_COLS = ["TMB", "weighted_aneuploidy", "cnv_breadth", "cnv_variance",
                    "subclonal_entropy", "clonality_score", "vaf_cv", "n_mutations"]
# driver-gene symbols for the mutation targets are kept OUT of the repo: supply
# them in a local (gitignored) data/driver_genes.txt, one per line. If absent,
# the mutation targets are simply skipped.
DRIVERS = []
_dg = os.path.join("data", "driver_genes.txt")
if os.path.exists(_dg):
    DRIVERS = [g.strip() for g in open(_dg) if g.strip()]
MIN_POS = 40        # min positives for a binary target
ALPHAS = np.logspace(-1, 4, 12)


def _read(omics_dir, fn):
    df = pd.read_parquet(os.path.join(omics_dir, fn))
    if df.index.name != "participant_id" and "participant_id" in df.columns:
        df = df.set_index("participant_id")
    return df[~df.index.duplicated(keep="first")]


def load_targets(omics_dir):
    """Return (Y dataframe patients x targets, meta list of (target, group, kind))."""
    cols, meta = [], []

    def add(df, names, group, kind):
        sub = df[[c for c in names if c in df.columns]].apply(pd.to_numeric, errors="coerce")
        for c in sub.columns:
            cols.append(sub[c].rename(c)); meta.append((c, group, kind))

    sig = _read(omics_dir, "IMMUNE_SIGNATURES.parquet")
    add(sig, [c for c in sig.columns if c.startswith("sig_")], "immune_signature", "cont")
    imm = _read(omics_dir, "IMMUNE_DECONVOLUTION.parquet")
    add(imm, [c for c in imm.columns if c.startswith("immune_")], "immune_fraction", "cont")
    add(_read(omics_dir, "PURITY_CONSOLIDATED.parquet"), PURITY_COLS, "purity", "cont")
    add(_read(omics_dir, "COMPOSITE_VOLATILITY_SCORES.parquet"), INSTABILITY_COLS,
        "instability", "cont")
    mut = _read(omics_dir, "MUTATIONS_FULL.parquet")
    add(mut, [f"bin_{g}" for g in DRIVERS], "mutation", "bin")

    Y = pd.concat(cols, axis=1)
    Y = Y.loc[:, ~Y.columns.duplicated()]
    return Y, meta


def within_cancer_r2(y, pred, canc):
    """R2 after removing per-cancer means from both y and pred."""
    df = pd.DataFrame({"y": y, "p": pred, "c": canc})
    yc = df["y"] - df.groupby("c")["y"].transform("mean")
    pc = df["p"] - df.groupby("c")["p"].transform("mean")
    ss_tot = float((yc ** 2).sum())
    return 1.0 - float(((yc - pc) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan


def within_cancer_auc(y, prob, canc):
    aucs, ws = [], []
    for c in np.unique(canc):
        m = canc == c
        if m.sum() >= 20 and 0 < y[m].sum() < m.sum():
            try:
                aucs.append(roc_auc_score(y[m], prob[m])); ws.append(int(m.sum()))
            except ValueError:
                pass
    return float(np.average(aucs, weights=ws)) if aucs else np.nan


def oof_predict(X, Y, kinds, cancers, random_state=42):
    """Out-of-fold predictions for every target, cancer-stratified 5-fold."""
    n, T = X.shape[0], Y.shape[1]
    pred = np.full((n, T), np.nan)
    skf = StratifiedKFold(5, shuffle=True, random_state=random_state)
    for tr, te in skf.split(X, cancers):
        scaler = StandardScaler().fit(X[tr])
        pca = PCA(n_components=0.95, random_state=random_state)
        Xtr = pca.fit_transform(scaler.transform(X[tr]))
        Xte = pca.transform(scaler.transform(X[te]))
        for j in range(T):
            ytr = Y[tr, j]
            ok = ~np.isnan(ytr)
            if ok.sum() < 30:
                continue
            if kinds[j] == "cont":
                m = RidgeCV(alphas=ALPHAS).fit(Xtr[ok], ytr[ok])
                pred[te, j] = m.predict(Xte)
            else:
                if 0 < ytr[ok].sum() < ok.sum():
                    m = LogisticRegression(max_iter=2000, class_weight="balanced",
                                           C=1.0).fit(Xtr[ok], ytr[ok].astype(int))
                    pred[te, j] = m.predict_proba(Xte)[:, 1]
    return pred


def make_figure():
    """Grouped bars of within-cancer predictability by group, mean vs std block."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(OUT)
    order = ["immune_signature", "instability", "purity", "immune_fraction", "mutation"]
    metric = {g: ("auc_within" if g == "mutation" else "r2_within") for g in order}
    piv = {b: {g: df[(df.block == b) & (df.group == g)][metric[g]].median()
               for g in order} for b in BLOCKS}
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(order))
    ax.barh(y + 0.2, [piv["mean"][g] for g in order], height=0.4, color="#0d9488",
            label="mean block")
    ax.barh(y - 0.2, [piv["std"][g] for g in order], height=0.4, color="#94a3b8",
            label="std block (heterogeneity)")
    ax.axvline(0, color="#9aa", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(["immune signatures (R²)", "genomic instability (R²)",
                        "tumor purity (R²)", "immune fractions (R²)",
                        "driver mutations (AUC)"])
    ax.set_xlabel("within-cancer predictive performance (median over targets)")
    ax.set_title("How much molecular state does the UNI2 embedding predict?",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    fig.tight_layout()
    out = os.path.join("figures", "uni2_predictability.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    import sys
    if "--figures-only" in sys.argv:
        make_figure(); return
    omics_dir = OMICS_DIR_DEFAULT
    clin = _read(omics_dir, "CLINICAL_ENHANCED.parquet")
    Y_all, meta = load_targets(omics_dir)
    names = [m[0] for m in meta]; groups = {m[0]: m[1] for m in meta}
    kinds_all = {m[0]: m[2] for m in meta}
    print(f"{len(names)} targets assembled", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for block in BLOCKS:
        print(f"\n=== block: {block} ===", flush=True)
        X = load_uni2_patient_matrix(block)
        cancer = clin["project_id"].reindex(X.index)
        common = X.index[cancer.notna().values].intersection(Y_all.index)
        Xc = X.loc[common]; canc = clin["project_id"].reindex(common).values
        Yc = Y_all.reindex(common)[names]
        kinds = [kinds_all[c] for c in names]
        pred = oof_predict(Xc.to_numpy(float), Yc.to_numpy(float), kinds, canc)

        for j, t in enumerate(names):
            y = Yc.iloc[:, j].to_numpy(float); p = pred[:, j]
            ok = ~np.isnan(y) & ~np.isnan(p)
            if ok.sum() < 50:
                continue
            yk, pk, ck = y[ok], p[ok], canc[ok]
            if kinds[j] == "cont":
                rows.append({"block": block, "group": groups[t], "target": t, "kind": "cont",
                             "n": int(ok.sum()),
                             "r2_pooled": round(float(r2_score(yk, pk)), 4),
                             "r2_within": round(within_cancer_r2(yk, pk, ck), 4)})
            else:
                yb = (yk > 0).astype(int)
                if not (0 < yb.sum() < len(yb)) or yb.sum() < MIN_POS:
                    continue
                rows.append({"block": block, "group": groups[t], "target": t, "kind": "bin",
                             "n": int(ok.sum()), "n_pos": int(yb.sum()),
                             "auc_pooled": round(float(roc_auc_score(yb, pk)), 4),
                             "auc_within": round(within_cancer_auc(yb, pk, ck), 4)})
        del X, Xc, Yc, pred; gc.collect(); thermal_pause(1.5)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(df)} rows)", flush=True)
    make_figure()
    # quick within-cancer summary by group/block
    for kind, col in (("cont", "r2_within"), ("bin", "auc_within")):
        sub = df[df["kind"] == kind]
        if len(sub):
            print(f"\nmedian {col} by group x block:")
            print(sub.pivot_table(col, "group", "block", "median").round(3).to_string())


if __name__ == "__main__":
    main()
