"""
aggregation_method_comparison.py
================================
Does the aggregation method matter for *prediction*, or only for PCA structure?

We re-aggregated all 8-cancer slides into results_v2/agg_full/full_methods.npz
(31 methods x 1,536 dims). So far that npz only feeds PCA pictures. This script
finally trains models on each method and ranks them, on two tasks:

  * Survival  — pan-cancer pooled, cancer-stratified 5-fold CV, CoxnetLasso on
                the WSI features. We report the honest within-cancer
                (stratified) C-index, plus the age+sex clinical baseline as a
                floor. Reuses run_pooled_cv from pooled_survival_analysis.py.
  * Grade     — low (G1/G2) vs high (G3/G4) across all graded cancers,
                GroupKFold by participant, PCA->classifier. Ranks methods by the
                best held-out AUC.

THERMAL-LIGHT FIRST PASS (by design):
  * no H5 access — the npz is already computed; this is sklearn on cached vectors
  * 5-fold CV (not 1,608-fit LOPO); bootstrap reduced to 200 for the CI
  * a 3-classifier subset for the grade sweep (fast, representative)
  * cooldown + gc between methods; survival CSV is checkpointed per method
    (re-run skips methods already written -> safe to stop/resume)
A full-rigor follow-up (LOPO, 1000 bootstrap, per-cancer, second WSI model) is
noted in the plan and meant to run overnight.

Run:  python aggregation_method_comparison.py
      python aggregation_method_comparison.py --bootstrap-iters 200 --cooldown 1.5
Output: results_v2/agg_full/survival_method_comparison.csv
        results_v2/agg_full/classification_grade_method_comparison.csv
"""
import os
import gc
import time
import argparse
import warnings

import numpy as np
import pandas as pd

from wsi_survival_pipeline import (
    _get_survival_models, _load_clinical_covariates, get_classifiers,
    thermal_pause,
)
from pooled_survival_analysis import run_pooled_cv

warnings.filterwarnings("ignore")

NPZ = os.path.join("results_v2", "agg_full", "full_methods.npz")
META = os.path.join("results_v2", "agg_full", "meta.csv")
CLINICAL = "CLINICAL_FULL.parquet"
OUT_DIR = os.path.join("results_v2", "agg_full")
SURV_CSV = os.path.join(OUT_DIR, "survival_method_comparison.csv")
CLASS_CSV = os.path.join(OUT_DIR, "classification_grade_method_comparison.csv")

# WSI survival model used to rank methods (lasso = sparse, robust, fast)
SWEEP_MODEL = "CoxnetLasso"
# fast + representative classifier subset for the grade sweep (both linear)
GRADE_CLASSIFIERS = ["LogReg_L2", "LDA"]
GRADE_MAP = {"G1": "low", "G2": "low", "G3": "high", "G4": "high"}
# fixed-rank PCA for the sweep: UNI2 features are high-rank, so PCA(0.95) keeps
# hundreds of comps and is slow. 50 comps preserves the method ranking at a
# fraction of the cost (full-rigor follow-up can restore PCA(0.95)).
SWEEP_PCA = 50


# ---------------------------------------------------------------- data loading
def load_npz():
    d = np.load(NPZ, allow_pickle=True)
    feats = d["features"]
    methods = [str(m) for m in d["method_names"]]
    bd = int(d["block_dim"])
    meta = pd.read_csv(META)
    print(f"Loaded npz: {feats.shape} | {len(methods)} methods x {bd} dims | "
          f"{len(meta)} slides", flush=True)
    return feats, methods, bd, meta


def _clinical_lookup(clinical_path):
    clin = pd.read_parquet(clinical_path)
    if "participant_id" not in clin.columns:
        clin.index.name = "participant_id"
        clin = clin.reset_index()
    return clin.drop_duplicates("participant_id").set_index("participant_id")


def build_method_cohort(block, meta, clin):
    """Per-patient (X, y, cancers, X_clin, pids) for one method's feature block.

    Mirrors load_pooled_cohort() in pooled_survival_analysis.py, but takes a
    feature block + meta instead of reading a CSV.
    """
    is_tumor = meta["sample_type"].values == "tumor"
    feats = block[is_tumor]
    pids_slide = meta["participant_id"].values[is_tumor]

    uniq = pd.unique(pids_slide)
    X_pat = np.vstack([feats[pids_slide == p].mean(axis=0) for p in uniq])

    time_s = clin["days_to_death"].fillna(clin["days_to_last_followup"])
    event_s = (clin["vital_status"].astype(str).str.lower() == "dead")
    project_s = clin["project_id"]

    keep, X_keep, t_keep, e_keep, c_keep = [], [], [], [], []
    for i, p in enumerate(uniq):
        if p in time_s.index and pd.notna(time_s.loc[p]) and float(time_s.loc[p]) > 0:
            keep.append(p)
            X_keep.append(X_pat[i])
            t_keep.append(float(time_s.loc[p]))
            e_keep.append(bool(event_s.loc[p]))
            c_keep.append(str(project_s.loc[p]))

    X = np.vstack(X_keep)
    y = np.array(list(zip(e_keep, t_keep)), dtype=[("event", bool), ("time", float)])
    cancers = np.array(c_keep)
    X_clin = _load_clinical_covariates(CLINICAL, keep)
    return X, y, cancers, X_clin, np.array(keep)


def build_grade_slidewise(meta, clin):
    """Per-SLIDE grade labels (low/high) for all graded tumor slides, any cancer.

    Returns (row_mask, labels, participant_ids) aligned to npz rows so any
    method block can be sliced the same way. Grade semantics differ across
    cancers — this ranks methods on a fixed label set, not a clinical claim.
    """
    grade = clin["tumor_grade"]
    is_tumor = meta["sample_type"].values == "tumor"
    rows, labels, pids = [], [], []
    for i in range(len(meta)):
        if not is_tumor[i]:
            continue
        p = meta["participant_id"].values[i]
        g = GRADE_MAP.get(grade.loc[p]) if p in grade.index else None
        if g is not None:
            rows.append(i); labels.append(g); pids.append(p)
    mask = np.zeros(len(meta), dtype=bool)
    mask[rows] = True
    return mask, labels, np.asarray(pids)


# ---------------------------------------------------------------- grade sweep
def quick_grade_auc(features, labels, pids, random_state=42, cooldown=0.3):
    """Best held-out AUC over a fast classifier subset (GroupKFold + PCA).

    Reuses the model definitions from get_classifiers() and the same
    StandardScaler->PCA(0.95) preprocessing as run_classification, restricted to
    GRADE_CLASSIFIERS to keep the sweep thermally light.
    """
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score

    y = LabelEncoder().fit_transform(labels)
    groups = np.asarray(pids)
    n_groups = len(pd.unique(groups))
    cv = GroupKFold(n_splits=min(5, n_groups))
    n_comp = min(SWEEP_PCA, features.shape[1], len(y) - 1)
    pre = Pipeline([("sc", StandardScaler()),
                    ("pca", PCA(n_components=n_comp, random_state=random_state))])
    allmodels = get_classifiers(n_jobs=1)

    best_auc, best_model = np.nan, None
    for name in GRADE_CLASSIFIERS:
        pipe = Pipeline([("pre", pre), ("clf", allmodels[name])])
        try:
            scores = None
            for m in ("predict_proba", "decision_function"):
                try:
                    out = cross_val_predict(pipe, features, y, cv=cv, groups=groups,
                                            method=m)
                    scores = out[:, 1] if out.ndim == 2 and out.shape[1] >= 2 else out
                    break
                except (AttributeError, ValueError):
                    continue
            if scores is not None:
                auc = roc_auc_score(y, scores)
                if np.isnan(best_auc) or auc > best_auc:
                    best_auc, best_model = float(auc), name
        except Exception as exc:
            print(f"      {name} failed: {exc}", flush=True)
        gc.collect()
        thermal_pause(cooldown)
    return best_auc, best_model


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-iters", type=int, default=200)
    ap.add_argument("--cooldown", type=float, default=1.5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="process only first N methods (smoke test)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    feats, methods, bd, meta = load_npz()
    if args.limit:
        methods = methods[:args.limit]
    clin = _clinical_lookup(CLINICAL)
    specs = _get_survival_models(args.random_state)

    # resume: skip methods already in the survival CSV
    done = set()
    if os.path.exists(SURV_CSV):
        prev = pd.read_csv(SURV_CSV)
        done = set(prev["method"].tolist())
        print(f"Resuming — {len(done)} survival rows already present", flush=True)

    # grade label set (identical across methods)
    gmask, glabels, gpids = build_grade_slidewise(meta, clin)
    print(f"Grade cohort: {gmask.sum()} tumor slides "
          f"({dict(pd.Series(glabels).value_counts())})", flush=True)

    surv_rows, class_rows = [], []
    t0 = time.time()
    baseline_done = "Cox_AgeSex_baseline" in done

    for mi, method in enumerate(methods):
        if method in done:
            continue
        print(f"\n[{mi+1}/{len(methods)}] {method}", flush=True)
        block = feats[:, mi * bd:(mi + 1) * bd]

        # --- survival
        X, y, cancers, X_clin, pids = build_method_cohort(block, meta, clin)
        metrics, _, _ = run_pooled_cv(
            X, y, cancers, X_clin, specs[SWEEP_MODEL], k=args.folds,
            random_state=args.random_state, n_bootstrap=args.bootstrap_iters,
            pca_components=SWEEP_PCA)
        metrics.pop("per_cancer_cindex", None)
        surv_rows.append({"method": method, "model": SWEEP_MODEL, **metrics})
        print(f"  survival: stratified C = {metrics['c_index_stratified']} | "
              f"overall C = {metrics['c_index_overall']} "
              f"[{metrics['ci_low']}, {metrics['ci_high']}]", flush=True)

        # clinical baseline once (method-independent; cohort is identical)
        if not baseline_done:
            bm, _, _ = run_pooled_cv(
                X, y, cancers, X_clin, specs["Cox_AgeSex"], k=args.folds,
                random_state=args.random_state, n_bootstrap=args.bootstrap_iters,
                pca_components=SWEEP_PCA)
            bm.pop("per_cancer_cindex", None)
            surv_rows.append({"method": "Cox_AgeSex_baseline",
                              "model": "Cox_AgeSex", **bm})
            baseline_done = True
            print(f"  baseline: stratified C = {bm['c_index_stratified']} "
                  f"(age+sex floor)", flush=True)

        # --- grade classification
        gfeat = block[gmask]
        auc, best = quick_grade_auc(gfeat, glabels, gpids,
                                    random_state=args.random_state)
        class_rows.append({"method": method, "best_grade_auc": round(auc, 4)
                           if not np.isnan(auc) else np.nan, "best_classifier": best,
                           "n_slides": int(gmask.sum())})
        print(f"  grade: best AUC = {auc:.4f} ({best})", flush=True)

        # checkpoint both CSVs after every method
        _append_csv(SURV_CSV, surv_rows); surv_rows = []
        _append_csv(CLASS_CSV, class_rows); class_rows = []
        gc.collect()
        thermal_pause(args.cooldown)

    dt = (time.time() - t0) / 60
    print(f"\nDONE in {dt:.1f} min. Wrote:\n  {SURV_CSV}\n  {CLASS_CSV}", flush=True)
    _print_ranking()


def _append_csv(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def _print_ranking():
    if os.path.exists(SURV_CSV):
        s = pd.read_csv(SURV_CSV)
        base = s[s["method"] == "Cox_AgeSex_baseline"]
        s = s[s["method"] != "Cox_AgeSex_baseline"].sort_values(
            "c_index_stratified", ascending=False)
        print("\nTop methods by stratified survival C-index:")
        for _, r in s.head(8).iterrows():
            print(f"  {r['method']:22s}  C = {r['c_index_stratified']}")
        if len(base):
            print(f"  {'(age+sex baseline)':22s}  C = "
                  f"{base.iloc[0]['c_index_stratified']}")


if __name__ == "__main__":
    main()
