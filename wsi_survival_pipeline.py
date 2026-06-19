#!/usr/bin/env python3
"""
WSI Survival Pipeline (consolidated, thermal-safe)
==================================================
One script for tumor/normal & grade & mutation classification AND
per-cancer pooled leave-one-participant-out (LOPO) survival analysis with
bootstrap 95% CIs on the C-index.

Pulls patch embeddings from one or more H5 directories
(default: ./embeddings + ./TCGA UNI2 embeddings), aggregates each slide to
a fixed-size vector (mean + std + percentiles), and runs:

  • classification CV (group-aware by participant) for tumor/normal /
    grade / mutation:GENE / survival_quartile targets
  • per-cancer survival LOPO + 1000-iter bootstrap CI for sksurv models
    (CoxnetLasso, CoxnetElasticNet, RSF, GradientBoostSurv)

LOPO is the right CV for survival here: with ~30–400 patients per cancer
and 5-fold C-index estimates that were essentially noise (e.g. RSF
0.71 ± 0.22), pooling held-out risk scores across all N participants and
computing a single C-index gives a stable estimate.

Usage
-----
    # Full run (classification + survival), both embedding dirs
    python wsi_survival_pipeline.py

    # Survival-only on the UNI2 cohort
    python wsi_survival_pipeline.py --no-classification \
        --embeddings-dir "./TCGA UNI2 embeddings"

    # Resume after a crash (uses checkpoint)
    python wsi_survival_pipeline.py --resume

    # Reuse pre-aggregated CSV (skip H5 loading entirely)
    python wsi_survival_pipeline.py --aggregated-csv ./results_v2/aggregated.csv

Thermal management: chunked H5 reads, per-file gc.collect(), cooldown
pauses between files, single-threaded models, checkpoint every 5 files.
"""

import argparse
import gc
import glob
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

COOLDOWN_NORMAL = 0.5
COOLDOWN_COOL = 2.0
H5_CHUNK_SIZE = 5000
DEFAULT_EMBEDDING_DIRS = ["./embeddings", "./TCGA UNI2 embeddings"]
BOOTSTRAP_ITERS = 1000


# ──────────────────────────────────────────────────────────────────────
# Thermal helpers
# ──────────────────────────────────────────────────────────────────────

def thermal_pause(seconds: float, verbose: bool = False) -> None:
    if seconds > 0:
        if verbose:
            print(f"    [cooldown {seconds:.1f}s]", end="", flush=True)
        time.sleep(seconds)
        if verbose:
            print(" ok")


# ──────────────────────────────────────────────────────────────────────
# H5 loading
# ──────────────────────────────────────────────────────────────────────

def load_h5_patches(h5_path: str, chunk_size: int = H5_CHUNK_SIZE) -> np.ndarray:
    """Load patch embeddings from one H5 file in chunks (thermal-safe)."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        dataset_key = None
        for key in ("features", "embeddings", "feats", "data"):
            if key in f:
                dataset_key = key
                break
        if dataset_key is None:
            dataset_key = list(f.keys())[0]

        dataset = f[dataset_key]
        shape = dataset.shape

        # (1, N, D) → (N, D)
        if len(shape) == 3 and shape[0] == 1:
            return np.array(dataset[0], dtype=np.float32)
        if len(shape) == 3:
            data = np.array(dataset, dtype=np.float32)
            return data.reshape(-1, shape[-1])

        n_patches = shape[0]
        if n_patches <= chunk_size:
            return np.array(dataset, dtype=np.float32)

        chunks = []
        for start in range(0, n_patches, chunk_size):
            chunks.append(np.array(dataset[start:start + chunk_size], dtype=np.float32))
        patches = np.vstack(chunks)
        del chunks
        gc.collect()
        return patches


def parse_tcga_barcode(filename: str) -> Tuple[str, str]:
    """Extract participant ID and sample type (tumor/normal) from TCGA barcode."""
    basename = os.path.basename(filename)
    match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})", basename)
    if not match:
        raise ValueError(f"Could not parse TCGA barcode from: {basename}")
    pid = match.group(1)
    sample_code = int(match.group(2))
    sample_type = "normal" if 10 <= sample_code <= 19 else "tumor"
    return pid, sample_type


def aggregate_patches(patches: np.ndarray, method: str = "rich") -> np.ndarray:
    """Aggregate (N_patches, D) → (D_agg,)."""
    if method == "mean":
        return np.mean(patches, axis=0).astype(np.float32)

    # 'rich' = mean + std + 4 percentiles → 6×D
    return np.concatenate([
        np.mean(patches, axis=0),
        np.std(patches, axis=0),
        np.percentile(patches, 10, axis=0),
        np.percentile(patches, 25, axis=0),
        np.percentile(patches, 75, axis=0),
        np.percentile(patches, 90, axis=0),
    ]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Embedding pipeline (multi-dir, checkpointed)
# ──────────────────────────────────────────────────────────────────────

def discover_h5_files(dirs: List[str]) -> List[str]:
    """Walk one or more directories and return sorted unique H5 paths."""
    seen = set()
    out = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.h5"))):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def load_and_aggregate(
    h5_dirs: List[str],
    method: str = "rich",
    cooldown: float = COOLDOWN_NORMAL,
    output_dir: str = "./results_v2",
    resume: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """Load + aggregate slide embeddings from one or more directories.

    Returns (features, sample_types, participant_ids, filenames).
    Saves checkpoint every 5 files so a crash never loses more than ~5 files.
    """
    h5_files = discover_h5_files(h5_dirs)
    if not h5_files:
        print(f"ERROR: no .h5 files found in {h5_dirs}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, ".aggregation_checkpoint.json")
    partial_npy = os.path.join(output_dir, ".partial_features.npy")

    features_list: List[np.ndarray] = []
    sample_types: List[str] = []
    participant_ids: List[str] = []
    filenames: List[str] = []
    completed: set = set()

    if resume and os.path.exists(ckpt_path) and os.path.exists(partial_npy):
        with open(ckpt_path) as fp:
            ck = json.load(fp)
        completed = set(ck.get("completed_files", []))
        sample_types = ck.get("sample_types", [])
        participant_ids = ck.get("participant_ids", [])
        filenames = ck.get("filenames", [])
        features_list = [row for row in np.load(partial_npy)]
        print(f"  Resuming: {len(completed)}/{len(h5_files)} files already done")

    skipped = []
    newly = 0
    for i, h5_path in enumerate(h5_files):
        basename = os.path.basename(h5_path)
        if basename in completed:
            continue
        try:
            pid, stype = parse_tcga_barcode(h5_path)
            patches = load_h5_patches(h5_path)
            slide_vec = aggregate_patches(patches, method=method)
            del patches
            gc.collect()

            features_list.append(slide_vec)
            sample_types.append(stype)
            participant_ids.append(pid)
            filenames.append(basename)
            completed.add(basename)
            newly += 1

            if verbose:
                print(f"  [{len(completed)}/{len(h5_files)}] {basename}: "
                      f"-> {slide_vec.shape[0]} feats, {stype}")

            if newly % 5 == 0:
                np.save(partial_npy, np.vstack(features_list))
                with open(ckpt_path, "w") as fp:
                    json.dump({
                        "completed_files": list(completed),
                        "sample_types": sample_types,
                        "participant_ids": participant_ids,
                        "filenames": filenames,
                    }, fp)

            thermal_pause(cooldown, verbose=(cooldown >= 1.0))

        except Exception as e:
            skipped.append((basename, str(e)))

    if skipped:
        print(f"\n  Skipped {len(skipped)} files (first 5):")
        for name, why in skipped[:5]:
            print(f"    {name}: {why}")

    features = np.vstack(features_list)

    # Clean up checkpoint on full success
    for f in (ckpt_path, partial_npy):
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    print(f"\n  Loaded {len(features)} slides — feature matrix: {features.shape}")
    return features, sample_types, participant_ids, filenames


def save_aggregated_csv(features: np.ndarray, sample_types: List[str],
                        participant_ids: List[str], filenames: List[str],
                        path: str) -> None:
    cols = [f"f{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=cols)
    df.insert(0, "filename", filenames)
    df.insert(1, "participant_id", participant_ids)
    df.insert(2, "sample_type", sample_types)
    df.to_csv(path, index=False)
    print(f"  Saved aggregated CSV: {path}")


def load_aggregated_csv(path: str) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    df = pd.read_csv(path)
    meta = {"filename", "participant_id", "sample_type", "label", "cancer_type"}
    feat_cols = [c for c in df.columns if c not in meta]
    features = df[feat_cols].values.astype(np.float32)
    # Accept either 'sample_type' (new schema) or 'label' (old schema)
    if "sample_type" in df.columns:
        sample_types = df["sample_type"].tolist()
    elif "label" in df.columns:
        sample_types = df["label"].tolist()
    else:
        sample_types = ["tumor"] * len(df)
    participant_ids = df["participant_id"].tolist()
    filenames = df.get("filename", pd.Series(participant_ids)).tolist()
    print(f"  Loaded {len(features)} rows, {features.shape[1]} features from {path}")
    return features, sample_types, participant_ids, filenames


# ──────────────────────────────────────────────────────────────────────
# Target relabeling
# ──────────────────────────────────────────────────────────────────────

def relabel_for_target(
    features: np.ndarray,
    sample_types: List[str],
    participant_ids: List[str],
    target: str,
    clinical_path: str,
    cancer_type: str = "CHOL",
    mutation_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Apply tumor/normal | grade | survival_quartile | mutation:GENE relabeling."""
    if target == "tumor_normal":
        return features, sample_types, participant_ids

    is_tumor = np.array([s == "tumor" for s in sample_types])
    features = features[is_tumor]
    pids = [p for p, t in zip(participant_ids, is_tumor) if t]

    clin = pd.read_parquet(clinical_path)
    if "participant_id" not in clin.columns:
        clin.index.name = "participant_id"
        clin = clin.reset_index()
    chol = clin[clin["project_id"] == f"TCGA-{cancer_type}"].set_index("participant_id")

    if target == "grade":
        gm = {"G1": "low", "G2": "low", "G3": "high", "G4": "high"}
        new_labels = [gm.get(chol.loc[p, "tumor_grade"]) if p in chol.index else None
                      for p in pids]

    elif target == "survival_quartile":
        times = {}
        for p in pids:
            if p not in chol.index:
                continue
            r = chol.loc[p]
            dead = str(r.get("vital_status", "")).lower() == "dead"
            t = r.get("days_to_death") if dead else r.get("days_to_last_followup")
            if pd.notna(t) and t > 0:
                times[p] = float(t)
        if len(times) < 8:
            raise ValueError(f"Too few patients with survival: {len(times)}")
        tvals = np.array(list(times.values()))
        q1, q3 = np.percentile(tvals, [25, 75])
        new_labels = []
        for p in pids:
            t = times.get(p)
            if t is None:
                new_labels.append(None)
            elif t <= q1:
                new_labels.append("short")
            elif t >= q3:
                new_labels.append("long")
            else:
                new_labels.append(None)

    elif target.startswith("mutation:"):
        gene = target.split(":", 1)[1].strip()
        if mutation_path is None:
            mutation_path = "./TCGA_data/MUTATIONS_FULL.parquet"
        mut = pd.read_parquet(mutation_path)
        col = f"mut_{gene}"
        if col not in mut.columns:
            raise ValueError(f"Gene '{gene}' not in mutation matrix (expected {col})")
        new_labels = [("mut" if int(mut.loc[p, col]) > 0 else "wt")
                      if p in mut.index else None for p in pids]
    else:
        raise ValueError(f"Unknown target: {target}")

    keep = [l is not None for l in new_labels]
    features_f = features[np.array(keep)]
    pids_f = [p for p, k in zip(pids, keep) if k]
    labels_f = [l for l in new_labels if l is not None]

    print(f"  Target '{target}': {len(features_f)} slides kept")
    print(f"  Label balance: {dict(pd.Series(labels_f).value_counts())}")
    return features_f, labels_f, pids_f


# ──────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────

def get_classifiers(n_jobs: int = 1) -> Dict:
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB

    return {
        "LogReg_L2": LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                        max_iter=2000, class_weight="balanced",
                                        random_state=42),
        "LogReg_L1": LogisticRegression(C=1.0, penalty="l1", solver="saga",
                                        max_iter=2000, class_weight="balanced",
                                        random_state=42),
        "RidgeClassifier": RidgeClassifier(alpha=1.0, class_weight="balanced"),
        "LinearSVC": LinearSVC(C=1.0, class_weight="balanced", max_iter=5000,
                               random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                               class_weight="balanced",
                                               random_state=42, n_jobs=n_jobs),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2,
                                           class_weight="balanced", random_state=42,
                                           n_jobs=n_jobs),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                      learning_rate=0.1,
                                                      random_state=42),
        "KNN_5": KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs),
        "LDA": LinearDiscriminantAnalysis(),
        "GaussianNB": GaussianNB(),
    }


def run_classification(
    features: np.ndarray,
    labels: List[str],
    participant_ids: List[str],
    n_folds: int = 5,
    pca_variance: float = 0.95,
    n_jobs: int = 1,
    cooldown: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Group-aware CV classification with PCA preprocessing inside each fold."""
    from sklearn.model_selection import (
        StratifiedKFold, GroupKFold, LeaveOneGroupOut, cross_val_predict,
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    groups = np.asarray(participant_ids)

    counts = pd.Series(groups).value_counts()
    paired = bool((counts > 1).any())
    n_unique = len(counts)

    if paired:
        cv = (LeaveOneGroupOut() if n_unique < 40
              else GroupKFold(n_splits=min(n_folds, n_unique)))
        cv_name = ("LeaveOneGroupOut" if n_unique < 40
                   else f"GroupKFold (k={cv.n_splits}, by participant)")
        cv_groups = groups
    else:
        cv = StratifiedKFold(n_splits=min(n_folds, len(y)), shuffle=True,
                             random_state=random_state)
        cv_name = f"StratifiedKFold (k={cv.n_splits})"
        cv_groups = None

    print(f"  Classes: {dict(zip(le.classes_, np.bincount(y)))}")
    print(f"  Features: {features.shape[1]} dims, PCA→{pca_variance:.0%} variance")
    print(f"  CV: {cv_name}")

    preprocess = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_variance, random_state=random_state)),
    ])

    results = []
    for name, model in get_classifiers(n_jobs=n_jobs).items():
        t0 = time.time()
        pipe = Pipeline([("preprocess", preprocess), ("clf", model)])
        try:
            preds = cross_val_predict(pipe, features, y, cv=cv, groups=cv_groups,
                                       method="predict")
            scores = None
            for m in ("predict_proba", "decision_function"):
                try:
                    out = cross_val_predict(pipe, features, y, cv=cv, groups=cv_groups,
                                             method=m)
                    scores = out[:, 1] if out.ndim == 2 and out.shape[1] >= 2 else out
                    break
                except (AttributeError, ValueError):
                    continue

            row = {"model": name,
                   "accuracy": round(accuracy_score(y, preds), 4)}
            if n_classes == 2:
                row["f1"] = round(f1_score(y, preds, average="binary",
                                            zero_division=0), 4)
                if scores is not None:
                    try:
                        row["auc"] = round(roc_auc_score(y, scores), 4)
                    except ValueError:
                        pass
            else:
                row["f1_macro"] = round(f1_score(y, preds, average="macro",
                                                  zero_division=0), 4)
                if scores is not None:
                    try:
                        row["auc_ovr"] = round(roc_auc_score(y, scores,
                                                              multi_class="ovr"), 4)
                    except ValueError:
                        pass
            row["time_sec"] = round(time.time() - t0, 2)
            results.append(row)
            auc = row.get("auc") or row.get("auc_ovr")
            print(f"  {name:20s}  Acc={row['accuracy']:.3f}  "
                  f"AUC={auc:.3f}" if auc else f"  {name:20s}  Acc={row['accuracy']:.3f}")
        except Exception as e:
            print(f"  {name:20s}  FAILED: {e}")
            results.append({"model": name, "error": str(e),
                            "time_sec": round(time.time() - t0, 2)})
        gc.collect()
        thermal_pause(cooldown)

    df = pd.DataFrame(results)
    sort_col = "auc" if "auc" in df.columns else "accuracy"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")
    return df


# ──────────────────────────────────────────────────────────────────────
# Survival: per-cancer pooled LOPO + bootstrap CI
# ──────────────────────────────────────────────────────────────────────

def _load_clinical_covariates(clinical_path: str,
                              pids: List[str]) -> np.ndarray:
    """Build per-patient covariate matrix aligned to `pids`.

    Returns (n, 2) array of [age_years, is_male]. Missing values are imputed
    with the cohort median (age) / mode (gender). `tumor_stage` is dropped
    from this parquet — it's NaN throughout — so the baseline is age+gender.
    """
    clin = pd.read_parquet(clinical_path)
    if "participant_id" not in clin.columns:
        clin.index.name = "participant_id"
        clin = clin.reset_index()
    df = clin.set_index("participant_id")

    age_days = pd.to_numeric(df["age_at_diagnosis"], errors="coerce")
    age_years = age_days / 365.25
    gender = df["gender"].astype(str).str.lower()
    is_male = pd.Series(np.where(gender == "male", 1.0,
                                  np.where(gender == "female", 0.0, np.nan)),
                         index=df.index)

    out = pd.DataFrame({"age_years": age_years, "is_male": is_male})
    out = out.reindex(pids)

    if out["age_years"].notna().any():
        out["age_years"] = out["age_years"].fillna(out["age_years"].median())
    else:
        out["age_years"] = 0.0
    if out["is_male"].notna().any():
        mode_val = float(out["is_male"].mode().iloc[0])
        out["is_male"] = out["is_male"].fillna(mode_val)
    else:
        out["is_male"] = 0.0

    return out.values.astype(np.float32)


def _get_survival_models(random_state: int = 42) -> Dict[str, Dict]:
    """Return survival model specs. Each value is {'model', 'feature_mode'}.

    feature_mode:
      'wsi'      — train on WSI embedding PCs only
      'clinical' — train on age + gender only (baseline Cox)
      'combined' — train on hstack(WSI PCs, clinical)
    """
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
    from sksurv.ensemble import (
        RandomSurvivalForest, GradientBoostingSurvivalAnalysis,
    )
    return {
        "Cox_AgeSex": {
            "model": CoxPHSurvivalAnalysis(alpha=1e-3),
            "feature_mode": "clinical",
        },
        "Cox_WSI_plus_Clin": {
            "model": CoxnetSurvivalAnalysis(
                l1_ratio=0.5, alpha_min_ratio=0.1, max_iter=1000),
            "feature_mode": "combined",
        },
        "CoxnetLasso": {
            "model": CoxnetSurvivalAnalysis(
                l1_ratio=1.0, alpha_min_ratio=0.1, max_iter=1000),
            "feature_mode": "wsi",
        },
        "CoxnetElasticNet": {
            "model": CoxnetSurvivalAnalysis(
                l1_ratio=0.5, alpha_min_ratio=0.1, max_iter=1000),
            "feature_mode": "wsi",
        },
        "RSF": {
            "model": RandomSurvivalForest(
                n_estimators=100, max_depth=5, min_samples_leaf=3,
                random_state=random_state, n_jobs=1),
            "feature_mode": "wsi",
        },
        "GradientBoostSurv": {
            "model": GradientBoostingSurvivalAnalysis(
                n_estimators=50, max_depth=2, learning_rate=0.1,
                random_state=random_state),
            "feature_mode": "wsi",
        },
    }


def _build_features(X_wsi: np.ndarray, X_clinical: Optional[np.ndarray],
                    mode: str) -> np.ndarray:
    if mode == "wsi":
        return X_wsi
    if mode == "clinical":
        if X_clinical is None:
            raise ValueError("feature_mode='clinical' requires X_clinical")
        return X_clinical
    if mode == "combined":
        if X_clinical is None:
            raise ValueError("feature_mode='combined' requires X_clinical")
        return np.hstack([X_wsi, X_clinical])
    raise ValueError(f"Unknown feature_mode: {mode}")


def _hazard_ratio_from_risk(risks: np.ndarray, times: np.ndarray,
                            events: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit lifelines Cox on the standardized risk score → (HR, low, high, p).

    HR is reported per 1-SD increase so it's comparable across models.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        return np.nan, np.nan, np.nan, np.nan
    sd = float(np.std(risks))
    if sd <= 0 or len(risks) < 5:
        return np.nan, np.nan, np.nan, np.nan
    z = (risks - np.mean(risks)) / sd
    df = pd.DataFrame({"z": z, "time": times, "event": events.astype(int)})
    try:
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(df, duration_col="time", event_col="event")
        hr = float(np.exp(cph.params_["z"]))
        ci = cph.confidence_intervals_.loc["z"]
        hr_low = float(np.exp(ci.iloc[0]))
        hr_high = float(np.exp(ci.iloc[1]))
        hr_p = float(cph.summary.loc["z", "p"])
        return hr, hr_low, hr_high, hr_p
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def _logrank_median_split(risks: np.ndarray, times: np.ndarray,
                          events: np.ndarray) -> Tuple[float, float]:
    """Log-rank test on median-split risk groups → (chi2, p)."""
    try:
        from sksurv.compare import compare_survival
    except ImportError:
        return np.nan, np.nan
    median_risk = float(np.median(risks))
    high = risks > median_risk
    if high.sum() < 5 or (~high).sum() < 5:
        return np.nan, np.nan
    y_arr = np.array(list(zip(events.astype(bool), times.astype(float))),
                      dtype=[("event", bool), ("time", float)])
    try:
        chi2, p = compare_survival(y_arr, high.astype(int))
        return float(chi2), float(p)
    except Exception:
        return np.nan, np.nan


def pooled_lopo_survival(
    X_wsi: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_factory,
    X_clinical: Optional[np.ndarray] = None,
    feature_mode: str = "wsi",
    n_bootstrap: int = BOOTSTRAP_ITERS,
    random_state: int = 42,
    cooldown: float = 0.0,
) -> Dict:
    """Pooled leave-one-participant-out C-index + bootstrap CI + HR + log-rank.

    Held-out predictions are collected across all participants and a single
    concordance_index_censored is computed on the full vector. Bootstrap
    resamples (event, time, risk) triples to get a 95% CI. Additionally
    reports per-SD hazard ratio (lifelines CoxPHFitter) and a median-split
    log-rank p-value.
    """
    from sksurv.metrics import concordance_index_censored

    unique_participants = np.unique(groups)
    n_participants = len(unique_participants)
    risk_scores = np.full(len(y), np.nan)

    rng = np.random.RandomState(random_state)

    for pid in unique_participants:
        test_mask = groups == pid
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            continue
        try:
            model = model_factory()
            X_tr = _build_features(
                X_wsi[train_mask],
                X_clinical[train_mask] if X_clinical is not None else None,
                feature_mode,
            )
            X_te = _build_features(
                X_wsi[test_mask],
                X_clinical[test_mask] if X_clinical is not None else None,
                feature_mode,
            )
            model.fit(X_tr, y[train_mask])
            pred = model.predict(X_te)
            risk_scores[test_mask] = pred
        except Exception:
            # Leave NaN for this participant; will be dropped before pooling
            pass
        if cooldown:
            thermal_pause(cooldown)

    nan_result = {
        "c_index_pooled": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "hr": np.nan, "hr_ci_low": np.nan, "hr_ci_high": np.nan,
        "hr_pvalue": np.nan, "logrank_chi2": np.nan, "logrank_pvalue": np.nan,
        "n_patients": n_participants, "n_events": int(y["event"].sum()),
        "n_valid": 0, "risk_scores": risk_scores,
    }
    valid = ~np.isnan(risk_scores)
    if valid.sum() < 2:
        nan_result["n_valid"] = int(valid.sum())
        return nan_result

    events = y["event"][valid]
    times = y["time"][valid]
    risks = risk_scores[valid]
    n_valid = int(valid.sum())

    c_pooled = concordance_index_censored(events, times, risks)[0]

    boot_cis = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_valid, n_valid)
        try:
            ci = concordance_index_censored(events[idx], times[idx], risks[idx])[0]
            boot_cis.append(ci)
        except Exception:
            continue
    if len(boot_cis) >= 100:
        ci_low, ci_high = np.percentile(boot_cis, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan

    hr, hr_low, hr_high, hr_p = _hazard_ratio_from_risk(risks, times, events)
    lr_chi2, lr_p = _logrank_median_split(risks, times, events)

    def _r(v, n=4):
        return round(float(v), n) if not np.isnan(v) else np.nan

    return {
        "c_index_pooled": _r(c_pooled),
        "ci_low": _r(ci_low), "ci_high": _r(ci_high),
        "hr": _r(hr), "hr_ci_low": _r(hr_low), "hr_ci_high": _r(hr_high),
        "hr_pvalue": _r(hr_p, 6),
        "logrank_chi2": _r(lr_chi2), "logrank_pvalue": _r(lr_p, 6),
        "n_patients": n_participants,
        "n_events": int(events.sum()),
        "n_valid": n_valid,
        "risk_scores": risk_scores,
    }


def run_survival_per_cancer(
    features: np.ndarray,
    participant_ids: List[str],
    sample_types: List[str],
    clinical_path: str,
    output_dir: str,
    min_patients: int = 30,
    pca_variance: float = 0.95,
    cooldown: float = 0.2,
    random_state: int = 42,
    n_bootstrap: int = BOOTSTRAP_ITERS,
) -> pd.DataFrame:
    """Per-cancer pooled LOPO survival across all TCGA projects in the cohort.

    Filters to tumor slides, matches each patient to clinical survival data,
    runs pooled LOPO + bootstrap CI for each (cancer, model) pair.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    # Tumor slides only
    is_tumor = np.array([s == "tumor" for s in sample_types])
    X_tumor = features[is_tumor]
    pids_tumor = [p for p, t in zip(participant_ids, is_tumor) if t]

    # Load clinical → derive (time, event) per participant
    clin = pd.read_parquet(clinical_path)
    if "participant_id" not in clin.columns:
        clin.index.name = "participant_id"
        clin = clin.reset_index()

    clin["time"] = clin["days_to_death"].fillna(clin["days_to_last_followup"])
    clin["event"] = (clin["vital_status"].str.lower() == "dead").astype(int)
    surv = clin.dropna(subset=["time"])
    surv = surv[surv["time"] > 0]

    pid_to_surv = surv.set_index("participant_id")[
        ["time", "event", "project_id"]].to_dict("index")

    # Restrict embedding rows to participants with valid survival
    keep_idx = [i for i, p in enumerate(pids_tumor) if p in pid_to_surv]
    X_all = X_tumor[keep_idx]
    pids_all = [pids_tumor[i] for i in keep_idx]
    cancers_all = np.array([pid_to_surv[p]["project_id"] for p in pids_all])

    print(f"\n  Survival cohort: {len(pids_all)} tumor slides "
          f"({len(set(pids_all))} unique participants) across "
          f"{len(np.unique(cancers_all))} cancer types")
    by_cancer = pd.Series(cancers_all).value_counts()
    print(f"  By cancer (slide count):\n{by_cancer.to_string()}")

    survival_models = _get_survival_models(random_state=random_state)

    risk_score_records = []  # for KM-plot reuse
    results = []

    for cancer in sorted(np.unique(cancers_all)):
        mask = cancers_all == cancer
        n_slides = int(mask.sum())
        unique_pids = np.unique([pids_all[i] for i in np.where(mask)[0]])
        n_pat = len(unique_pids)
        if n_pat < min_patients:
            print(f"\n── {cancer} skipped (n={n_pat} < {min_patients}) ──")
            continue

        X_c = X_all[mask]
        pids_c = np.array([pids_all[i] for i in np.where(mask)[0]])
        # Per-patient (time, event); a participant may have >1 slide but
        # survival is per-patient — broadcast their (time,event) to each slide
        y_c = np.array(
            [(bool(pid_to_surv[p]["event"]), float(pid_to_surv[p]["time"]))
             for p in pids_c],
            dtype=[("event", bool), ("time", float)],
        )

        # PCA refit on this cancer subset
        pca_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=random_state)),
        ])
        X_c_pca = pca_pipe.fit_transform(X_c)

        # Standardize clinical covariates separately so they sit on the
        # same scale as the PCs when combined.
        X_clin_raw = _load_clinical_covariates(clinical_path, pids_c.tolist())
        clin_scaler = StandardScaler()
        X_clin = clin_scaler.fit_transform(X_clin_raw)

        n_events = int(y_c["event"].sum())
        print(f"\n── {cancer}: {n_pat} patients, {n_slides} slides, "
              f"{n_events} events | PCA: {X_c.shape[1]} → {X_c_pca.shape[1]} | "
              f"clinical: age+gender ──")

        for name, spec in survival_models.items():
            t0 = time.time()
            model_template = spec["model"]
            feature_mode = spec["feature_mode"]

            def factory(_m=model_template):
                from sklearn.base import clone
                try:
                    return clone(_m)
                except Exception:
                    return type(_m)(**_m.get_params())

            try:
                res = pooled_lopo_survival(
                    X_c_pca, y_c, pids_c, factory,
                    X_clinical=X_clin,
                    feature_mode=feature_mode,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state, cooldown=0.0,
                )
                elapsed = time.time() - t0
                row = {
                    "cancer_type": cancer,
                    "model": name,
                    "feature_mode": feature_mode,
                    "clinical_covariates_used": ("age,gender"
                        if feature_mode in ("clinical", "combined") else "none"),
                    "n_patients": n_pat,
                    "n_slides": n_slides,
                    "n_events": n_events,
                    "c_index_pooled": res["c_index_pooled"],
                    "ci_low": res["ci_low"],
                    "ci_high": res["ci_high"],
                    "hr": res["hr"],
                    "hr_ci_low": res["hr_ci_low"],
                    "hr_ci_high": res["hr_ci_high"],
                    "hr_pvalue": res["hr_pvalue"],
                    "logrank_chi2": res["logrank_chi2"],
                    "logrank_pvalue": res["logrank_pvalue"],
                    "n_valid": res["n_valid"],
                    "time_sec": round(elapsed, 2),
                }
                results.append(row)
                rs = res["risk_scores"]
                for i, (p, r) in enumerate(zip(pids_c, rs)):
                    risk_score_records.append({
                        "cancer_type": cancer, "model": name,
                        "feature_mode": feature_mode,
                        "participant_id": p, "risk_score": r,
                        "time": y_c["time"][i], "event": int(y_c["event"][i]),
                    })

                c, lo, hi = row["c_index_pooled"], row["ci_low"], row["ci_high"]
                hr_str = (f" HR={row['hr']:.2f} p={row['hr_pvalue']:.1e}"
                          if not np.isnan(row["hr"]) else "")
                lr_str = (f" logrank-p={row['logrank_pvalue']:.1e}"
                          if not np.isnan(row["logrank_pvalue"]) else "")
                print(f"  {name:20s}  C={c:.3f} [{lo:.3f}, {hi:.3f}]"
                      f"{hr_str}{lr_str}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  {name:20s}  FAILED: {e}")
                results.append({
                    "cancer_type": cancer, "model": name,
                    "feature_mode": feature_mode,
                    "n_patients": n_pat, "n_events": n_events,
                    "c_index_pooled": np.nan, "error": str(e),
                })
            gc.collect()
            thermal_pause(cooldown)

    df = pd.DataFrame(results)
    os.makedirs(os.path.join(output_dir, "survival_loo"), exist_ok=True)
    out_csv = os.path.join(output_dir, "survival_loo", "all_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    if risk_score_records:
        rs_df = pd.DataFrame(risk_score_records)
        rs_csv = os.path.join(output_dir, "survival_loo", "risk_scores.csv")
        rs_df.to_csv(rs_csv, index=False)
        print(f"  Saved per-slide risk scores: {rs_csv}")

    return df


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    p = argparse.ArgumentParser(
        description="WSI Survival Pipeline (consolidated, thermal-safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--embeddings-dir", action="append", default=None,
                   help="Embeddings directory (pass multiple times to combine). "
                        f"Default: {DEFAULT_EMBEDDING_DIRS}")
    p.add_argument("--clinical-data", default="./CLINICAL_FULL.parquet")
    p.add_argument("--mutation-data", default="./TCGA_data/MUTATIONS_FULL.parquet")
    p.add_argument("--output-dir", default="./results_v2")
    p.add_argument("--aggregated-csv", default=None,
                   help="Skip H5 loading and reuse a saved aggregated CSV")
    p.add_argument("--aggregation", choices=["mean", "rich"], default="rich")

    # Targets
    p.add_argument("--target", default="tumor_normal",
                   help="tumor_normal | grade | survival_quartile | mutation:GENE")
    p.add_argument("--cancer-type", default="CHOL",
                   help="Cancer type for grade/mutation/quartile relabeling")

    # Survival
    p.add_argument("--no-survival", action="store_true")
    p.add_argument("--no-classification", action="store_true")
    p.add_argument("--min-survival-patients", type=int, default=30)
    p.add_argument("--bootstrap-iters", type=int, default=BOOTSTRAP_ITERS)

    # CV
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)

    # Thermal
    p.add_argument("--cool", action="store_true",
                   help="Aggressive thermal mode")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--resume", action="store_true")

    args = p.parse_args()

    cooldown = COOLDOWN_COOL if args.cool else COOLDOWN_NORMAL
    n_jobs = 1 if args.cool else args.n_jobs

    print("=" * 70)
    print("WSI Survival Pipeline" + (" (COOL MODE)" if args.cool else ""))
    print("=" * 70)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load embeddings ──
    if args.aggregated_csv:
        print(f"\nLoading pre-aggregated features: {args.aggregated_csv}")
        features, sample_types, participant_ids, filenames = \
            load_aggregated_csv(args.aggregated_csv)
    else:
        dirs = args.embeddings_dir or [d for d in DEFAULT_EMBEDDING_DIRS
                                       if os.path.isdir(d)]
        print(f"\nLoading embeddings from: {dirs}")
        features, sample_types, participant_ids, filenames = \
            load_and_aggregate(
                dirs, method=args.aggregation,
                cooldown=cooldown, output_dir=args.output_dir,
                resume=args.resume,
            )
        agg_path = os.path.join(args.output_dir, "aggregated_embeddings.csv")
        save_aggregated_csv(features, sample_types, participant_ids,
                             filenames, agg_path)

    # ── Classification ──
    if not args.no_classification:
        print(f"\n{'─' * 70}\nCLASSIFICATION ({args.target})\n{'─' * 70}")
        feats_c, labels_c, pids_c = relabel_for_target(
            features, sample_types, participant_ids,
            target=args.target,
            clinical_path=args.clinical_data,
            cancer_type=args.cancer_type,
            mutation_path=args.mutation_data,
        )
        if len(set(labels_c)) < 2:
            print("  Only one class after filtering — skipping classification.")
        else:
            clf_df = run_classification(
                feats_c, labels_c, pids_c,
                n_folds=args.n_folds, n_jobs=n_jobs,
                cooldown=cooldown * 0.4, random_state=args.random_state,
            )
            tag = args.target.replace(":", "_")
            out_csv = os.path.join(args.output_dir,
                                    f"classification_{tag}.csv")
            clf_df.to_csv(out_csv, index=False)
            print(f"\n  Saved: {out_csv}")

    # ── Survival (pooled LOPO + bootstrap) ──
    if not args.no_survival and os.path.exists(args.clinical_data):
        print(f"\n{'─' * 70}\nSURVIVAL (pooled LOPO + bootstrap CI)\n{'─' * 70}")
        try:
            run_survival_per_cancer(
                features, participant_ids, sample_types,
                clinical_path=args.clinical_data,
                output_dir=args.output_dir,
                min_patients=args.min_survival_patients,
                cooldown=cooldown * 0.4,
                random_state=args.random_state,
                n_bootstrap=args.bootstrap_iters,
            )
        except ImportError as e:
            print(f"  scikit-survival missing — skipping survival ({e})")

    print(f"\n{'=' * 70}\nAll results saved to: {args.output_dir}/\nDone!")


if __name__ == "__main__":
    main()
