"""
pooled_survival_analysis.py
===========================
Pan-cancer POOLED survival analysis: train one model across ALL 8 TCGA cohorts
together, instead of one model per cancer.

Why two C-indices are reported
-------------------------------
Naively pooling cancers inflates the concordance index, because different
cancers have very different baseline survival — a model can score well just by
sorting patients by *cancer type* rather than by within-patient risk. So we
report:

  * c_index_overall    — concordance over ALL patients pooled (optimistic)
  * c_index_stratified — concordance computed only between patients of the SAME
                         cancer, then pair-weighted across cancers (the honest
                         within-cancer signal)

Method: features collapsed to one vector per patient, StandardScaler->PCA(0.95)
refit inside each fold, cancer-stratified 5-fold CV (ample events at n~1600, far
lighter than 1600 LOPO fits). Reuses model specs + HR/log-rank helpers from
wsi_survival_pipeline.py.

Run:  python pooled_survival_analysis.py
      python pooled_survival_analysis.py --aggregated-csv results_v2/aggregated_embeddings.csv \
             --clinical-data CLINICAL_FULL.parquet --output-dir results_v2
"""
import os
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sksurv.metrics import concordance_index_censored

from wsi_survival_pipeline import (
    load_aggregated_csv, _get_survival_models, _load_clinical_covariates,
    _build_features, _hazard_ratio_from_risk, _logrank_median_split,
)

warnings.filterwarnings("ignore")

# the trio that matters: clinical floor, WSI-only, and combined
POOLED_MODELS = ["Cox_AgeSex", "CoxnetLasso", "Cox_WSI_plus_Clin"]


def load_pooled_cohort(aggregated_csv, clinical_path):
    """Return per-patient (X, y, cancers, X_clin, pids) pooled across cancers."""
    features, sample_types, participant_ids, _ = load_aggregated_csv(aggregated_csv)
    features = np.asarray(features)
    sample_types = np.asarray(sample_types)
    participant_ids = np.asarray(participant_ids)

    is_tumor = sample_types == "tumor"
    features = features[is_tumor]
    participant_ids = participant_ids[is_tumor]

    # collapse to one feature vector per patient (mean over their slides)
    uniq = pd.unique(participant_ids)
    X_pat = np.vstack([features[participant_ids == p].mean(axis=0) for p in uniq])
    pids = list(uniq)

    # survival labels + cancer type from clinical
    clin = pd.read_parquet(clinical_path)
    if "participant_id" not in clin.columns:
        clin.index.name = "participant_id"
        clin = clin.reset_index()
    clin = clin.drop_duplicates("participant_id").set_index("participant_id")
    time = clin["days_to_death"].fillna(clin["days_to_last_followup"])
    event = (clin["vital_status"].astype(str).str.lower() == "dead")
    project = clin["project_id"]

    keep, X_keep, t_keep, e_keep, c_keep = [], [], [], [], []
    for i, p in enumerate(pids):
        if p in time.index and pd.notna(time.loc[p]) and float(time.loc[p]) > 0:
            keep.append(p)
            X_keep.append(X_pat[i])
            t_keep.append(float(time.loc[p]))
            e_keep.append(bool(event.loc[p]))
            c_keep.append(str(project.loc[p]))

    X = np.vstack(X_keep)
    y = np.array(list(zip(e_keep, t_keep)), dtype=[("event", bool), ("time", float)])
    cancers = np.array(c_keep)
    X_clin = _load_clinical_covariates(clinical_path, keep)
    return X, y, cancers, X_clin, np.array(keep)


def run_pooled_cv(X, y, cancers, X_clin, model_spec, k=5, random_state=42,
                  n_bootstrap=1000, pca_components=0.95):
    """Cancer-stratified k-fold CV → held-out risk per patient + metrics.

    `pca_components` is passed straight to PCA's `n_components` (default 0.95 =
    95% variance, the original behavior). Pass an int (e.g. 50) for a faster,
    fixed-rank reduction — useful when sweeping many feature sets.
    """
    feature_mode = model_spec["feature_mode"]
    base_model = model_spec["model"]
    n = len(y)
    risk = np.full(n, np.nan)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    for tr, te in skf.split(X, cancers):
        # refit scaler+PCA on the WSI features of the TRAIN fold only
        if feature_mode in ("wsi", "combined"):
            pca = Pipeline([("sc", StandardScaler()),
                            ("pca", PCA(n_components=pca_components, random_state=random_state))])
            Xw_tr = pca.fit_transform(X[tr])
            Xw_te = pca.transform(X[te])
        else:
            Xw_tr = X[tr]; Xw_te = X[te]  # unused for clinical mode

        if X_clin is not None:
            cs = StandardScaler().fit(X_clin[tr])
            Xc_tr = cs.transform(X_clin[tr]); Xc_te = cs.transform(X_clin[te])
        else:
            Xc_tr = Xc_te = None

        try:
            model = clone(base_model)
            model.fit(_build_features(Xw_tr, Xc_tr, feature_mode), y[tr])
            risk[te] = model.predict(_build_features(Xw_te, Xc_te, feature_mode))
        except Exception as exc:
            print(f"    fold failed: {exc}")

    valid = ~np.isnan(risk)
    events = y["event"][valid]; times = y["time"][valid]; risks = risk[valid]
    canc = cancers[valid]
    nv = int(valid.sum())

    # overall pooled C-index + bootstrap CI
    c_overall = concordance_index_censored(events, times, risks)[0]
    rng = np.random.RandomState(random_state)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, nv, nv)
        try:
            boot.append(concordance_index_censored(events[idx], times[idx], risks[idx])[0])
        except Exception:
            pass
    ci_low, ci_high = (np.percentile(boot, [2.5, 97.5]) if len(boot) >= 100
                       else (np.nan, np.nan))

    # stratified C-index: within-cancer concordance, pair-weighted
    num = den = 0.0
    per_cancer = {}
    for c in np.unique(canc):
        m = canc == c
        if m.sum() < 5 or events[m].sum() < 1:
            continue
        try:
            res = concordance_index_censored(events[m], times[m], risks[m])
            cidx, conc, disc, tied_r = res[0], res[1], res[2], res[3]
            pairs = conc + disc + tied_r
            if pairs > 0:
                num += cidx * pairs; den += pairs
                per_cancer[c] = round(float(cidx), 4)
        except Exception:
            pass
    c_strat = num / den if den > 0 else np.nan

    hr, hr_lo, hr_hi, hr_p = _hazard_ratio_from_risk(risks, times, events)
    lr_chi2, lr_p = _logrank_median_split(risks, times, events)

    def r(v, n=4):
        return round(float(v), n) if v is not None and not np.isnan(v) else np.nan

    metrics = {
        "feature_mode": feature_mode,
        "n_patients": nv, "n_events": int(events.sum()),
        "c_index_overall": r(c_overall),
        "ci_low": r(ci_low), "ci_high": r(ci_high),
        "c_index_stratified": r(c_strat),
        "hr": r(hr), "hr_ci_low": r(hr_lo), "hr_ci_high": r(hr_hi),
        "hr_pvalue": r(hr_p, 6),
        "logrank_chi2": r(lr_chi2), "logrank_pvalue": r(lr_p, 6),
        "per_cancer_cindex": per_cancer,
    }
    return metrics, risk, valid


def plot_pooled_km(risk, y, model_name, out_path):
    """Median-split KM of the pooled held-out risk for one model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    valid = ~np.isnan(risk)
    r = risk[valid]; t = y["time"][valid] / 365.25; e = y["event"][valid]
    high = r > np.median(r)
    lr = logrank_test(t[high], t[~high], e[high], e[~high])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    kmf = KaplanMeierFitter()
    for mask, color, lab in [(~high, "#0d9488", "low predicted risk"),
                             (high, "#c2410c", "high predicted risk")]:
        kmf.fit(t[mask], e[mask], label=f"{lab} (n={int(mask.sum())})")
        kmf.plot_survival_function(ax=ax, color=color, ci_alpha=0.12)
    ax.set_title(f"Pan-cancer pooled model ({model_name})\n"
                 f"all 8 cancers, held-out risk · log-rank p = {lr.p_value:.1e}")
    ax.set_xlabel("years since diagnosis"); ax.set_ylabel("survival probability")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path, f"(log-rank p={lr.p_value:.2e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregated-csv", default="results_v2/aggregated_embeddings.csv")
    ap.add_argument("--clinical-data", default="CLINICAL_FULL.parquet")
    ap.add_argument("--output-dir", default="results_v2")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.join(args.output_dir, "survival_pooled")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading pooled cohort ...")
    X, y, cancers, X_clin, pids = load_pooled_cohort(args.aggregated_csv, args.clinical_data)
    print(f"  {len(y)} patients · {int(y['event'].sum())} events · "
          f"{len(np.unique(cancers))} cancers · {X.shape[1]} features")
    for c in sorted(np.unique(cancers)):
        m = cancers == c
        print(f"    {c}: {int(m.sum())} patients, {int(y['event'][m].sum())} events")

    specs = _get_survival_models(args.random_state)
    rows, risk_rows = [], []
    for name in POOLED_MODELS:
        print(f"\nModel: {name}")
        metrics, risk, valid = run_pooled_cv(
            X, y, cancers, X_clin, specs[name], k=args.folds,
            random_state=args.random_state, n_bootstrap=args.bootstrap_iters)
        per_cancer = metrics.pop("per_cancer_cindex")
        print(f"  overall C = {metrics['c_index_overall']} "
              f"[{metrics['ci_low']}, {metrics['ci_high']}] · "
              f"stratified C = {metrics['c_index_stratified']} · "
              f"HR/SD = {metrics['hr']} · logrank p = {metrics['logrank_pvalue']}")
        print(f"  per-cancer C: {per_cancer}")
        rows.append({"model": name, **metrics})
        for i in np.where(valid)[0]:
            risk_rows.append({"model": name, "participant_id": pids[i],
                              "cancer_type": cancers[i], "risk_score": float(risk[i]),
                              "time": float(y["time"][i]), "event": int(y["event"][i])})

    res = pd.DataFrame(rows)
    res_path = os.path.join(out_dir, "all_results.csv")
    res.to_csv(res_path, index=False)
    print(f"\nwrote {res_path}")
    pd.DataFrame(risk_rows).to_csv(os.path.join(out_dir, "risk_scores_pooled.csv"), index=False)
    print(f"wrote {os.path.join(out_dir, 'risk_scores_pooled.csv')}")

    # KM for the strongest WSI-informed model by stratified C-index
    wsi_rows = res[res["model"] != "Cox_AgeSex"]
    best = wsi_rows.loc[wsi_rows["c_index_stratified"].idxmax(), "model"]
    best_idx = POOLED_MODELS.index(best)
    _, best_risk, _ = run_pooled_cv(X, y, cancers, X_clin, specs[best],
                                    k=args.folds, random_state=args.random_state,
                                    n_bootstrap=0)
    plot_pooled_km(best_risk, y, best, os.path.join("figures", "km_pancancer_pooled.png"))


if __name__ == "__main__":
    main()
