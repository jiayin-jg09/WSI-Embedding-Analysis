"""
uni2_interactions.py
====================
Do PAIRS of UNI2 dimensions jointly predict survival or tumor type beyond either
alone? We screen every dimension pair, calibrate against a permutation null, then
cross-validate whether the top interactions actually ADD over the main effects.

Why scaling matters (and what we do): a raw product X*Y of uncentered variables is
mostly a restatement of the main effects and collinear with them. We therefore
CENTER + STANDARDIZE each variable before forming the product (z-score), fit on the
TRAIN fold only. The screen below assumes standardized inputs and a standardized,
mean-0 target.

Feasibility: C(1536,2) = 1.18M pairs (a 7.5 GB matrix if materialized). We never
build it — screening every pair vs a target is three 1536x1536 matmuls:
  A = (Z*y).T @ Z   (sum z_i z_j y)
  B = Z.T @ Z       (sum z_i z_j)
  C = (Z^2).T @(Z^2)(sum z_i^2 z_j^2)
  corr(z_i*z_j, y) = (A/n) / sqrt(C/n - (B/n)^2)      # y standardized, mean~0
Off-diagonal = interactions; diagonal = quadratic main effects.

Outcomes:
  survival  -> within-cancer: target = per-cancer martingale residuals; dims
               residualized on cancer dummies (so the screen is within-cancer)
  tumor     -> 8-class: per-class one-vs-rest target, aggregated by max |corr|

Validation (top-N): cancer-stratified 5-fold; does (z_i, z_j, z_i*z_j) beat
(z_i, z_j) on held-out C-index (survival) / AUC (tumor)?

Names are hidden: dims are plain numbers. Output CSVs are gitignored; figures
carry no feature/program names. Run:
  python uni2_interactions.py --outcome both --perms 200 --top-n 150
"""
import os
import gc
import argparse
import warnings
import numpy as np
import pandas as pd

from wsi_survival_pipeline import thermal_pause
from uni2_omics_correlation import load_uni2_patient_matrix, OMICS_DIR_DEFAULT, residualize

warnings.filterwarnings("ignore")
DIR = os.path.join("results_v2", "dim_omics")
CLIN = os.path.join(OMICS_DIR_DEFAULT, "CLINICAL_ENHANCED.parquet")


def zscore(M):
    M = M - M.mean(0)
    sd = M.std(0); sd[sd == 0] = 1.0
    return M / sd


def load_patients(agg="mean"):
    """Per-patient dim matrix + cancer + (time,event), aligned."""
    X = load_uni2_patient_matrix(agg)        # index participant_id, tumor slides collapsed
    clin = pd.read_parquet(CLIN)
    if clin.index.name != "participant_id" and "participant_id" in clin.columns:
        clin = clin.set_index("participant_id")
    clin = clin[~clin.index.duplicated(keep="first")]
    common = X.index.intersection(clin.index)
    X = X.loc[common]
    c = clin.loc[common]
    cancer = c["project_id"].astype(str).values
    time = pd.to_numeric(c.get("OS.time"), errors="coerce").values
    event = pd.to_numeric(c.get("OS"), errors="coerce").values
    return X, cancer, time, event


def martingale_resid(time, event, cancer):
    """Null martingale residuals computed per cancer (Nelson-Aalen baseline)."""
    from lifelines import NelsonAalenFitter
    y = np.full(len(time), np.nan)
    for cc in np.unique(cancer):
        m = cancer == cc
        t, e = time[m], event[m]
        ok = np.isfinite(t) & np.isfinite(e) & (t > 0)
        if ok.sum() < 5:
            continue
        naf = NelsonAalenFitter().fit(t[ok], e[ok])
        H = naf.cumulative_hazard_at_times(t[ok]).values.ravel()
        r = np.full(m.sum(), np.nan); idx = np.where(ok)[0]
        r[idx] = e[ok] - H
        y[m] = r
    return y


def screen(Z, y):
    """corr of every product z_i*z_j with standardized y -> (d x d) matrix."""
    n = Z.shape[0]
    A = (Z * y[:, None]).T @ Z
    B = Z.T @ Z
    C = (Z ** 2).T @ (Z ** 2)
    var = np.maximum(C / n - (B / n) ** 2, 1e-9)
    return (A / n) / np.sqrt(var)            # sd(y)=1, mean(y)~0


def offdiag_top(corr, k):
    d = corr.shape[0]
    iu = np.triu_indices(d, k=1)
    vals = corr[iu]
    order = np.argsort(np.abs(vals))[::-1][:k]
    return [(int(iu[0][o]), int(iu[1][o]), float(vals[o])) for o in order]


def perm_null(Z, y, inv, perms, rng):
    """Family-wise null: max |off-diag corr| over shuffles of y."""
    n, d = Z.shape
    iu = np.triu_indices(d, k=1)
    maxes = []
    for _ in range(perms):
        yp = y[rng.permutation(n)]
        A = (Z * yp[:, None]).T @ Z
        corr = (A / n) * inv
        maxes.append(float(np.abs(corr[iu]).max()))
    return np.array(maxes)


# ---------------------------------------------------------------- validation
def cv_delta_survival(xi, xj, time, event, cancer, k=5, seed=42):
    """Held-out C-index of Cox(i,j,ij) minus Cox(i,j), cancer-stratified."""
    from sklearn.model_selection import StratifiedKFold
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    ok = np.isfinite(time) & np.isfinite(event) & (time > 0)
    xi, xj, t, e, cc = xi[ok], xj[ok], time[ok], event[ok], cancer[ok]
    pred_main = np.full(len(t), np.nan); pred_int = np.full(len(t), np.nan)
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for tr, te in skf.split(xi, cc):
        def z(a, tridx, teidx):
            mu, sd = a[tridx].mean(), a[tridx].std() or 1.0
            return (a[tridx] - mu) / sd, (a[teidx] - mu) / sd
        zi_tr, zi_te = z(xi, tr, te); zj_tr, zj_te = z(xj, tr, te)
        pr_tr = zi_tr * zj_tr; mu, sd = pr_tr.mean(), pr_tr.std() or 1.0
        pr_tr = (pr_tr - mu) / sd; pr_te = ((zi_te * zj_te) - mu) / sd
        base = pd.DataFrame({"zi": zi_tr, "zj": zj_tr, "T": t[tr], "E": e[tr]})
        try:
            m1 = CoxPHFitter(penalizer=0.1).fit(base, "T", "E")
            pred_main[te] = m1.predict_partial_hazard(
                pd.DataFrame({"zi": zi_te, "zj": zj_te})).values
            bi = base.assign(ij=pr_tr)
            m2 = CoxPHFitter(penalizer=0.1).fit(bi, "T", "E")
            pred_int[te] = m2.predict_partial_hazard(
                pd.DataFrame({"zi": zi_te, "zj": zj_te, "ij": pr_te})).values
        except Exception:
            return np.nan
    ok2 = np.isfinite(pred_main) & np.isfinite(pred_int)
    if ok2.sum() < 30:
        return np.nan
    c_main = concordance_index(t[ok2], -pred_main[ok2], e[ok2])
    c_int = concordance_index(t[ok2], -pred_int[ok2], e[ok2])
    return round(c_int - c_main, 4)


def cv_delta_tumor(xi, xj, cancer, k=5, seed=42):
    """Held-out macro-OVR AUC of multiclass logistic with vs without interaction."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    try:
        ncls = len(np.unique(cancer))
        ylab = LabelEncoder().fit_transform(cancer)
        skf = StratifiedKFold(k, shuffle=True, random_state=seed)
        pm = np.zeros((len(ylab), ncls)); pi = pm.copy()
        for tr, te in skf.split(xi, ylab):
            if len(np.unique(ylab[tr])) < ncls:
                return np.nan                      # a class missing from a train fold
            def z(a):
                mu, sd = a[tr].mean(), a[tr].std() or 1.0
                return (a[tr] - mu) / sd, (a[te] - mu) / sd
            zi_tr, zi_te = z(xi); zj_tr, zj_te = z(xj)
            Xtr = np.c_[zi_tr, zj_tr]; Xte = np.c_[zi_te, zj_te]
            ij_tr = zi_tr * zj_tr; ij_te = zi_te * zj_te
            lr = LogisticRegression(max_iter=500)
            pm[te] = lr.fit(Xtr, ylab[tr]).predict_proba(Xte)
            pi[te] = lr.fit(np.c_[Xtr, ij_tr], ylab[tr]).predict_proba(np.c_[Xte, ij_te])
        a_main = roc_auc_score(ylab, pm, multi_class="ovr", average="macro")
        a_int = roc_auc_score(ylab, pi, multi_class="ovr", average="macro")
        return round(a_int - a_main, 4)
    except Exception:
        return np.nan


# ---------------------------------------------------------------- main
def run_outcome(outcome, Xraw, cancer, time, event, args, rng):
    n, d = Xraw.shape
    Zc = zscore(Xraw)
    if outcome == "survival":
        y = martingale_resid(time, event, cancer)
        keep = np.isfinite(y)
        Z = residualize(Zc[keep], pd.get_dummies(cancer[keep]).to_numpy(float))
        Z = zscore(Z); y = zscore(y[keep].reshape(-1, 1)).ravel()
        Xv, cancv, tv, ev = Xraw[keep], cancer[keep], time[keep], event[keep]
        corr = screen(Z, y)
    else:  # tumor: per-class OVR, aggregate max|corr|
        Z = Zc
        cls = np.unique(cancer)
        agg = np.zeros((d, d))
        for cc in cls:
            yc = zscore((cancer == cc).astype(float).reshape(-1, 1)).ravel()
            agg = np.maximum(agg, np.abs(screen(Z, yc)))
        corr = agg
        Xv, cancv, tv, ev = Xraw, cancer, time, event
    # permutation null (family-wise max |off-diag corr|)
    n2 = Z.shape[0]
    B = Z.T @ Z; C = (Z ** 2).T @ (Z ** 2)
    inv = 1.0 / np.sqrt(np.maximum(C / n2 - (B / n2) ** 2, 1e-9))   # 1/sd(product)
    if outcome == "survival":
        nulls = perm_null(Z, y, inv, args.perms, rng)
    else:
        nulls = perm_null_tumor(Z, cancer, inv, args.perms, rng)
    thr = float(np.percentile(nulls, 95))
    top = offdiag_top(corr, args.top_n)
    print(f"[{outcome}] screen done | perm95 thr={thr:.3f} | "
          f"{sum(abs(v) > thr for _,_,v in top)} of top-{args.top_n} beat null", flush=True)

    rows = []
    for i, j, v in top:
        if outcome == "survival":
            dlt = cv_delta_survival(Xv[:, i], Xv[:, j], tv, ev, cancv)
        else:
            dlt = cv_delta_tumor(Xv[:, i], Xv[:, j], cancv)
        rows.append({"dim_i": i, "dim_j": j, "screen_corr": round(v, 4),
                     "beats_null": bool(abs(v) > thr), "delta_metric": dlt})
    out = pd.DataFrame(rows).sort_values("delta_metric", ascending=False, na_position="last")
    p = os.path.join(DIR, f"interactions_{outcome}_dimdim.csv")
    out.to_csv(p, index=False)
    npos = int((out["delta_metric"] > 0).sum())
    print(f"[{outcome}] wrote {p} | {npos}/{len(out)} top pairs improve held-out metric", flush=True)
    return corr, nulls, out


def perm_null_tumor(Z, cancer, inv, perms, rng):
    n, d = Z.shape; iu = np.triu_indices(d, k=1); cls = np.unique(cancer); maxes = []
    for _ in range(perms):
        cp = cancer[rng.permutation(n)]
        agg = np.zeros((d, d))
        for cc in cls:
            yc = zscore((cp == cc).astype(float).reshape(-1, 1)).ravel()
            A = (Z * yc[:, None]).T @ Z
            agg = np.maximum(agg, np.abs((A / n) * inv))
        maxes.append(float(agg[iu].max()))
    return np.array(maxes)


def screen_cross(F, D, y):
    """corr of every cross product f_a*d_j with y -> (n_feat x n_dim)."""
    n = F.shape[0]
    A = (F * y[:, None]).T @ D
    B = F.T @ D
    C = (F ** 2).T @ (D ** 2)
    var = np.maximum(C / n - (B / n) ** 2, 1e-9)
    return (A / n) / np.sqrt(var)


def run_featdim(omic, outcome, X, cancer, time, event, args, rng):
    """feature(omic) x dimension interactions for one outcome."""
    from uni2_omics_correlation import load_omic, OMIC_SPECS
    import hashlib
    Y = load_omic(omic, OMICS_DIR_DEFAULT)
    common = X.index.intersection(Y.index)
    Xd = X.loc[common]; Yf = Y.loc[common]
    # align outcome arrays to `common` order via Series (intersection may reorder)
    cser = pd.Series(cancer, index=X.index).loc[common]
    tser = pd.Series(time, index=X.index).loc[common]
    eser = pd.Series(event, index=X.index).loc[common]
    Yf = Yf.loc[:, Yf.notna().any(axis=0)]
    keepp = (Yf.notna().mean(axis=1) >= 0.5).values
    Xd, Yf = Xd[keepp], Yf[keepp]
    Yf = Yf.fillna(Yf.median(axis=0)).fillna(0.0)   # 0-fill columns all-NaN after filtering
    if OMIC_SPECS[omic][2] == "variable":
        Yf = Yf[Yf.var(axis=0).sort_values(ascending=False).head(2000).index]
    canc = cser.values[keepp]; t = tser.values[keepp]; e = eser.values[keepp]
    D = zscore(Xd.to_numpy(float)); F = zscore(Yf.to_numpy(float))
    feats = list(Yf.columns)
    ab = {"expression": "EXPR", "cnv": "CNV", "immune_signatures": "SIG",
          "rppa": "RPPA", "immune": "IMM", "mutations": "MUT"}.get(omic, "X")

    def hcode(name):
        return f"{ab}-" + hashlib.sha1(str(name).upper().encode()).hexdigest()[:5].upper()

    if outcome == "survival":
        y = martingale_resid(t, e, canc); ok = np.isfinite(y)
        Dd = zscore(residualize(D[ok], pd.get_dummies(canc[ok]).to_numpy(float)))
        Ff = zscore(residualize(F[ok], pd.get_dummies(canc[ok]).to_numpy(float)))
        y = zscore(y[ok].reshape(-1, 1)).ravel()
        corr = screen_cross(Ff, Dd, y); Fv, Dv, cv, tv, ev = F[ok], D[ok], canc[ok], t[ok], e[ok]
        nF, nD = Ff.shape[1], Dd.shape[1]
        B = Ff.T @ Dd; C = (Ff ** 2).T @ (Dd ** 2)
        inv = 1.0 / np.sqrt(np.maximum(C / Ff.shape[0] - (B / Ff.shape[0]) ** 2, 1e-9))
        nulls = []
        for _ in range(args.perms):
            yp = y[rng.permutation(len(y))]
            cp = (((Ff * yp[:, None]).T @ Dd) / Ff.shape[0]) * inv
            nulls.append(float(np.abs(cp).max()))
        nulls = np.array(nulls)
    else:
        Dd, Ff = D, F; cls = np.unique(canc)
        corr = np.zeros((F.shape[1], D.shape[1]))
        for cc in cls:
            yc = zscore((canc == cc).astype(float).reshape(-1, 1)).ravel()
            corr = np.maximum(corr, np.abs(screen_cross(Ff, Dd, yc)))
        Fv, Dv, cv, tv, ev = F, D, canc, t, e
        B = Ff.T @ Dd; C = (Ff ** 2).T @ (Dd ** 2)
        inv = 1.0 / np.sqrt(np.maximum(C / Ff.shape[0] - (B / Ff.shape[0]) ** 2, 1e-9))
        nulls = []
        for _ in range(args.perms):
            cpan = canc[rng.permutation(len(canc))]; agg = np.zeros_like(corr)
            for cc in cls:
                yc = zscore((cpan == cc).astype(float).reshape(-1, 1)).ravel()
                agg = np.maximum(agg, np.abs((((Ff * yc[:, None]).T @ Dd) / Ff.shape[0]) * inv))
            nulls.append(float(agg.max()))
        nulls = np.array(nulls)

    thr = float(np.percentile(nulls, 95))
    flat = np.abs(corr).ravel()
    order = np.argsort(flat)[::-1][:args.top_n]
    rows = []
    for o in order:
        a, j = divmod(int(o), corr.shape[1])
        v = float(corr[a, j])
        if outcome == "survival":
            dlt = cv_delta_survival(Fv[:, a], Dv[:, j], tv, ev, cv)
        else:
            dlt = cv_delta_tumor(Fv[:, a], Dv[:, j], cv)
        rows.append({"feature_code": hcode(feats[a]), "dim_j": j, "screen_corr": round(v, 4),
                     "beats_null": bool(abs(v) > thr), "delta_metric": dlt})
    out = pd.DataFrame(rows).sort_values("delta_metric", ascending=False, na_position="last")
    p = os.path.join(DIR, f"interactions_{outcome}_featdim_{omic}.csv")
    out.to_csv(p, index=False)
    print(f"[{outcome}/{omic}] thr={thr:.3f} | "
          f"{int(out.beats_null.sum())}/{len(out)} beat null | "
          f"{int((out.delta_metric>0).sum())}/{len(out)} improve held-out "
          f"(max {out.delta_metric.max():.4f}) -> {p}", flush=True)
    gc.collect(); thermal_pause(args.cooldown)


def make_compare_figure():
    """Name-free summary: do cross-modality feature-with-dimension pairs beat
    same-space dimension-with-dimension pairs? Reads the committed-analysis CSVs
    (gitignored) and emits one figure. Modality labels only -- no feature names."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

    # (label, csv-suffix). dim x dim is the same-space baseline.
    sources = [
        ("dim x dim", "dimdim"),
        ("feature x dim\n(expression)", "featdim_expression"),
        ("feature x dim\n(immune sig.)", "featdim_immune_signatures"),
        ("feature x dim\n(proteomic)", "featdim_rppa"),
    ]

    def load(outcome, suffix):
        p = os.path.join(DIR, f"interactions_{outcome}_{suffix}.csv")
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p)
        return dict(beats=int(d.beats_null.sum()), n=len(d),
                    med_delta=float(np.nanmedian(d.delta_metric)))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#0d9488", "#94a3b8", "#94a3b8", "#94a3b8"]

    # Panel A -- survival: pairs surviving the family-wise permutation null.
    sa = [load("survival", s) for _, s in sources]
    labs = [lab for lab, _ in sources]
    beats = [r["beats"] if r else 0 for r in sa]
    bars = axA.bar(range(len(labs)), beats, color=colors)
    for b, v in zip(bars, beats):
        axA.text(b.get_x() + b.get_width() / 2, v + 0.4, str(v),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    axA.set_xticks(range(len(labs))); axA.set_xticklabels(labs, fontsize=8)
    axA.set_ylabel("pairs beating permutation null  (of 150)")
    axA.set_title("Survival: only same-space pairs survive the null", fontweight="bold", fontsize=10)
    axA.set_ylim(0, max(beats) + 3 if max(beats) else 3)

    # Panel B -- tumor: interactions screen significant but add ~nothing on held-out AUC.
    sb = [load("tumor", s) for _, s in sources]
    # proteomic tumor is degenerate (sparse measurement -> empty OVR null); drop it, note it.
    keep = [(lab, r) for (lab, _), r in zip(sources, sb)
            if r and r["n"] and not np.isnan(r["med_delta"])]
    klabs = [lab for lab, _ in keep]
    deltas = [r["med_delta"] for _, r in keep]
    barsB = axB.bar(range(len(klabs)), deltas, color=colors[:len(klabs)])
    for b, v in zip(barsB, deltas):
        axB.text(b.get_x() + b.get_width() / 2, v, f"{v:+.4f}",
                 ha="center", va="bottom", fontsize=9)
    axB.set_xticks(range(len(klabs))); axB.set_xticklabels(klabs, fontsize=8)
    axB.set_ylabel("median held-out AUC gain from interaction term")
    axB.set_title("Tumor type: pairs significant but add ~0 over main effects",
                  fontweight="bold", fontsize=10)
    axB.axhline(0, color="#475569", lw=0.8)

    fig.suptitle("Pairwise interaction screen: feature x dim vs dim x dim",
                 fontweight="bold")
    fig.text(0.5, 0.005,
             "Proteomic tumor panel omitted (sparse measurement yields a degenerate null). "
             "Counts/gains are within-cancer, cancer-stratified 5-fold.",
             ha="center", fontsize=7.5, color="#475569")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = os.path.join("figures", "interactions_featdim_vs_dimdim.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", choices=["survival", "tumor", "both"], default="both")
    ap.add_argument("--source", choices=["dimdim", "featdim"], default="dimdim")
    ap.add_argument("--omics", nargs="*", default=["immune_signatures", "rppa", "expression"])
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--top-n", type=int, default=150)
    ap.add_argument("--cooldown", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare-fig", action="store_true",
                    help="just regenerate the feature x dim vs dim x dim summary figure from CSVs")
    args = ap.parse_args()
    os.makedirs(DIR, exist_ok=True)
    if args.compare_fig:
        make_compare_figure()
        return
    rng = np.random.default_rng(args.seed)

    X, cancer, time, event = load_patients("mean")
    Xraw = X.to_numpy(float)
    print(f"patients={Xraw.shape[0]} dims={Xraw.shape[1]} "
          f"cancers={len(np.unique(cancer))} events={int(np.nansum(event))}", flush=True)
    outcomes = ["survival", "tumor"] if args.outcome == "both" else [args.outcome]

    if args.source == "featdim":
        for omic in args.omics:
            for oc in outcomes:
                run_featdim(omic, oc, X, cancer, time, event, args, rng)
        return

    results = {}
    for oc in outcomes:
        results[oc] = run_outcome(oc, Xraw, cancer, time, event, args, rng)
        gc.collect(); thermal_pause(args.cooldown)
    make_figures(results)


def make_figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    for oc, (corr, nulls, out) in results.items():
        # null vs observed-top distribution
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.hist(nulls, bins=30, color="#94a3b8", label="permutation max |corr|")
        thr = np.percentile(nulls, 95)
        ax.axvline(thr, color="#c2410c", ls="--", label=f"95th pct = {thr:.2f}")
        ax.axvline(out["screen_corr"].abs().max(), color="#0d9488",
                   label=f"top observed = {out['screen_corr'].abs().max():.2f}")
        ax.set_title(f"{oc}: interaction screen vs permutation null", fontweight="bold")
        ax.set_xlabel("|correlation| of dimension-pair product with outcome"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join("figures", f"interactions_{oc}_null.png"),
                                        dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"wrote figures/interactions_{oc}_null.png", flush=True)


if __name__ == "__main__":
    main()
