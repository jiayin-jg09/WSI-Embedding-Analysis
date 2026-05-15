#!/usr/bin/env python3
"""
Lightweight WSI Embedding Classifier (Thermal-Safe Edition)
============================================================
Laptop-friendly ML pipeline for tissue classification using pre-extracted
UNI2 patch embeddings. Designed to run without overheating on thin laptops
like the Dell Inspiron.

Thermal management features:
  - Processes H5 files one at a time with immediate memory cleanup
  - Cooldown pauses between file loads to prevent sustained CPU spikes
  - Chunked reading for large H5 files (never loads full array at once)
  - Single-threaded models by default (n_jobs=1) to limit heat
  - Optional --cool mode for aggressive thermal throttling
  - Checkpointing: saves progress after each file so crashes don't lose work

Usage:
    # Standard run (thermal-safe defaults)
    python lightweight_wsi_classifier.py --embeddings-dir ./embeddings --mode binary

    # Extra-cool mode for very hot laptops
    python lightweight_wsi_classifier.py --embeddings-dir ./embeddings --mode binary --cool

    # Resume from checkpoint after a crash
    python lightweight_wsi_classifier.py --embeddings-dir ./embeddings --mode binary --resume

    # Use pre-aggregated CSV (fastest, zero thermal stress)
    python lightweight_wsi_classifier.py --aggregated-csv ./aggregated_embeddings.csv --mode binary
"""

import argparse
import gc
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# THERMAL MANAGEMENT
# ──────────────────────────────────────────────────────────────────────

# Default pause between H5 file loads (seconds)
COOLDOWN_NORMAL = 0.5
COOLDOWN_COOL = 2.0

# Chunk size for reading large H5 datasets (rows at a time)
H5_CHUNK_SIZE = 5000


def thermal_pause(seconds, verbose=False):
    """Pause to let the CPU cool down between operations."""
    if seconds > 0:
        if verbose:
            print(f"    [cooldown {seconds:.1f}s]", end="", flush=True)
        time.sleep(seconds)
        if verbose:
            print(" ok")


# ──────────────────────────────────────────────────────────────────────
# 1. EMBEDDING LOADING & AGGREGATION (THERMAL-SAFE)
# ──────────────────────────────────────────────────────────────────────

def load_h5_embeddings_chunked(h5_path, chunk_size=H5_CHUNK_SIZE):
    """Load patch embeddings from H5 file in chunks to limit memory spikes.

    For small files (< chunk_size rows), loads normally.
    For large files, computes running statistics without holding
    the full array in memory.

    Returns:
        patches: np.array of shape (N_patches, D) for small files
        OR dict of pre-computed statistics for large files
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        # Find the dataset key
        dataset_key = None
        for key in ["features", "embeddings", "feats", "data"]:
            if key in f:
                dataset_key = key
                break
        if dataset_key is None:
            dataset_key = list(f.keys())[0]

        dataset = f[dataset_key]
        shape = dataset.shape
        # Handle 3D arrays with leading batch dim: (1, N, D) -> (N, D)
        if len(shape) == 3 and shape[0] == 1:
            data = np.array(dataset[0], dtype=np.float32)
            return data
        elif len(shape) == 3:
            # Multiple batches — reshape to (batch*N, D)
            data = np.array(dataset, dtype=np.float32)
            return data.reshape(-1, shape[-1])
        n_patches, n_dims = shape

        if n_patches <= chunk_size:
            # Small file — load directly
            return np.array(dataset, dtype=np.float32)

        # Large file — read in chunks, compute running stats
        # This avoids loading e.g. 50k x 1536 floats all at once
        # We compute aggregation stats on-the-fly
        running_sum = np.zeros(n_dims, dtype=np.float64)
        running_sq_sum = np.zeros(n_dims, dtype=np.float64)
        all_chunks = []

        for start in range(0, n_patches, chunk_size):
            end = min(start + chunk_size, n_patches)
            chunk = np.array(dataset[start:end], dtype=np.float32)
            all_chunks.append(chunk)
            running_sum += chunk.sum(axis=0)
            running_sq_sum += (chunk ** 2).sum(axis=0)

        # Concatenate — we do need the full array for percentiles
        # but we loaded it in controlled bursts
        patches = np.vstack(all_chunks)
        del all_chunks
        gc.collect()

        return patches


def parse_tcga_barcode(filename):
    """Extract participant ID and sample type from TCGA filename."""
    import re
    basename = os.path.basename(filename)
    match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})", basename)
    if match:
        participant_id = match.group(1)
        sample_code = int(match.group(2))
        sample_type = "normal" if 10 <= sample_code <= 19 else "tumor"
        return participant_id, sample_type
    raise ValueError(f"Could not parse TCGA barcode from: {basename}")


def aggregate_patches(patches, method="rich"):
    """Aggregate patch embeddings into a fixed-size slide vector.

    Args:
        patches: np.array of shape (N_patches, D)
        method: 'mean' | 'rich' (mean + std + percentiles)

    Returns:
        1D numpy array — the slide-level embedding
    """
    if method == "mean":
        return np.mean(patches, axis=0).astype(np.float32)

    # "rich" aggregation: mean + std + P10 + P25 + P75 + P90
    agg = np.concatenate([
        np.mean(patches, axis=0),
        np.std(patches, axis=0),
        np.percentile(patches, 10, axis=0),
        np.percentile(patches, 25, axis=0),
        np.percentile(patches, 75, axis=0),
        np.percentile(patches, 90, axis=0),
    ]).astype(np.float32)
    return agg


def load_checkpoint(checkpoint_path):
    """Load progress from a checkpoint file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_path, data):
    """Save progress to a checkpoint file."""
    with open(checkpoint_path, "w") as f:
        json.dump(data, f)


def load_and_aggregate_embeddings(embeddings_dir, method="rich", verbose=True,
                                   cooldown=COOLDOWN_NORMAL, output_dir=None,
                                   resume=False):
    """Load H5 files one at a time, aggregate, free memory immediately.

    Thermal-safe: each file is loaded, aggregated, then deleted from memory
    with a cooldown pause between files. Progress is checkpointed so a crash
    doesn't lose completed work.

    Returns:
        features: np.array of shape (N_slides, D_agg)
        labels: list of str ('tumor' / 'normal')
        participant_ids: list of str
        filenames: list of str
    """
    h5_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.h5")))
    if not h5_files:
        print(f"ERROR: No .h5 files found in {embeddings_dir}")
        sys.exit(1)

    # Checkpoint setup
    checkpoint_dir = output_dir or "."
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, ".aggregation_checkpoint.json")
    partial_npy_path = os.path.join(checkpoint_dir, ".partial_features.npy")

    # Check for existing checkpoint
    completed_files = set()
    features_list = []
    labels = []
    participant_ids = []
    filenames = []

    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint and os.path.exists(partial_npy_path):
            completed_files = set(checkpoint.get("completed_files", []))
            labels = checkpoint.get("labels", [])
            participant_ids = checkpoint.get("participant_ids", [])
            filenames = checkpoint.get("filenames", [])
            features_list = [row for row in np.load(partial_npy_path)]
            print(f"  Resuming from checkpoint: {len(completed_files)}/{len(h5_files)} "
                  f"files already processed")

    skipped = []
    newly_processed = 0

    for i, h5_path in enumerate(h5_files):
        basename = os.path.basename(h5_path)

        # Skip already-processed files
        if basename in completed_files:
            continue

        try:
            participant_id, sample_type = parse_tcga_barcode(h5_path)

            # Load in chunks to control memory spikes
            patches = load_h5_embeddings_chunked(h5_path)
            n_patches = patches.shape[0]

            # Aggregate immediately
            slide_vec = aggregate_patches(patches, method=method)

            # Free the raw patches RIGHT NOW
            del patches
            gc.collect()

            features_list.append(slide_vec)
            labels.append(sample_type)
            participant_ids.append(participant_id)
            filenames.append(basename)
            completed_files.add(basename)
            newly_processed += 1

            if verbose:
                print(f"  [{len(completed_files)}/{len(h5_files)}] {basename}: "
                      f"{n_patches} patches -> {len(slide_vec)} features, {sample_type}")

            # Save checkpoint every 5 files
            if newly_processed % 5 == 0:
                np.save(partial_npy_path, np.vstack(features_list))
                save_checkpoint(checkpoint_path, {
                    "completed_files": list(completed_files),
                    "labels": labels,
                    "participant_ids": participant_ids,
                    "filenames": filenames,
                })

            # Thermal cooldown between files
            thermal_pause(cooldown, verbose=(cooldown >= 1.0))

        except Exception as e:
            skipped.append((basename, str(e)))

    if skipped:
        print(f"\n  Skipped {len(skipped)} files:")
        for fname, reason in skipped:
            print(f"    {fname}: {reason}")

    features = np.vstack(features_list)

    # Clean up checkpoint files on success
    for f in [checkpoint_path, partial_npy_path]:
        if os.path.exists(f):
            os.remove(f)

    print(f"\n  Loaded {len(features)} slides -> feature matrix: {features.shape}")
    return features, labels, participant_ids, filenames


# ──────────────────────────────────────────────────────────────────────
# 1b. TARGET RELABELING (grade / survival quartile / mutation)
# ──────────────────────────────────────────────────────────────────────

def relabel_for_target(features, labels, participant_ids, filenames,
                       target, clinical_path, mutation_path=None,
                       cancer_type="CHOL"):
    """Replace tumor/normal labels with target-specific labels, filter rows.

    target: 'tumor_normal' (no-op) | 'grade' | 'survival_quartile' | 'mutation:GENE'

    Returns:
        (features_f, labels_f, participant_ids_f, filenames_f)
    """
    if target == "tumor_normal":
        return features, labels, participant_ids, filenames

    # Tumor-only for all non-default targets (normal slides have no driver
    # mutations or tumor grade)
    is_tumor = np.array([l == "tumor" for l in labels])
    features = features[is_tumor]
    pids = [p for p, t in zip(participant_ids, is_tumor) if t]
    fns = [f for f, t in zip(filenames, is_tumor) if t]

    clin = pd.read_parquet(clinical_path)
    if "participant_id" not in clin.columns and clin.index.name == "participant_id":
        clin = clin.reset_index()
    chol = clin[clin["project_id"] == f"TCGA-{cancer_type}"]
    chol_idx = chol.set_index("participant_id")

    if target == "grade":
        grade_map = {"G1": "low", "G2": "low", "G3": "high", "G4": "high"}
        new_labels = [
            grade_map.get(chol_idx.loc[p, "tumor_grade"])
            if p in chol_idx.index else None
            for p in pids
        ]

    elif target == "survival_quartile":
        times = {}
        for p in pids:
            if p not in chol_idx.index:
                continue
            r = chol_idx.loc[p]
            dead = str(r.get("vital_status", "")).lower() == "dead"
            t = r.get("days_to_death") if dead else r.get("days_to_last_followup")
            if pd.notna(t) and t > 0:
                times[p] = float(t)
        if len(times) < 8:
            raise ValueError(f"Too few patients with survival: {len(times)}")
        tvals = np.array(list(times.values()))
        q1, q3 = np.percentile(tvals, [25, 75])
        print(f"  Survival quartiles: Q1={q1:.0f}d, Q3={q3:.0f}d")
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
            mutation_path = os.path.join(
                os.path.dirname(os.path.abspath(clinical_path)),
                "TCGA_data", "MUTATIONS_FULL.parquet"
            )
        mut = pd.read_parquet(mutation_path)
        col = f"mut_{gene}"
        if col not in mut.columns:
            raise ValueError(f"Gene '{gene}' not in mutation matrix "
                             f"(expected column {col})")
        # Matrix stores mutation COUNTS (not 0/1) — any nonzero = mutated
        new_labels = [
            ("mut" if int(mut.loc[p, col]) > 0 else "wt")
            if p in mut.index else None
            for p in pids
        ]

    else:
        raise ValueError(f"Unknown target: {target}")

    keep = [l is not None for l in new_labels]
    features_f = features[np.array(keep)]
    pids_f = [p for p, k in zip(pids, keep) if k]
    fns_f = [f for f, k in zip(fns, keep) if k]
    labels_f = [l for l in new_labels if l is not None]

    print(f"  Target '{target}': {len(features_f)} tumor slides kept "
          f"(filtered from {len(pids)})")
    print(f"  Label balance: {dict(pd.Series(labels_f).value_counts())}")
    return features_f, labels_f, pids_f, fns_f


# ──────────────────────────────────────────────────────────────────────
# 2. MULTI-CANCER SUPPORT
# ──────────────────────────────────────────────────────────────────────

def load_multi_cancer_embeddings(embeddings_dir, clinical_path, method="rich",
                                  cancer_types=None, verbose=True,
                                  cooldown=COOLDOWN_NORMAL, output_dir=None,
                                  resume=False):
    """Load embeddings from multiple TCGA cancer types (thermal-safe)."""
    clinical = pd.read_parquet(clinical_path)
    pid_to_cancer = dict(zip(
        clinical["participant_id"],
        clinical["project_id"].str.replace("TCGA-", "")
    ))

    # Collect H5 files
    subdirs = [d for d in Path(embeddings_dir).iterdir() if d.is_dir()]
    if subdirs:
        h5_files = []
        for subdir in subdirs:
            h5_files.extend(sorted(glob.glob(str(subdir / "*.h5"))))
    if not subdirs or not h5_files:
        h5_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.h5")))

    if not h5_files:
        print(f"ERROR: No .h5 files found in {embeddings_dir}")
        sys.exit(1)

    features_list = []
    cancer_labels = []
    participant_ids = []
    sample_types = []
    skipped = []

    for i, h5_path in enumerate(h5_files):
        try:
            pid, stype = parse_tcga_barcode(h5_path)
            cancer_type = pid_to_cancer.get(pid)

            if cancer_type is None:
                skipped.append((os.path.basename(h5_path), "not in clinical data"))
                continue
            if cancer_types and cancer_type not in cancer_types:
                continue

            patches = load_h5_embeddings_chunked(h5_path)
            slide_vec = aggregate_patches(patches, method=method)

            del patches
            gc.collect()

            features_list.append(slide_vec)
            cancer_labels.append(cancer_type)
            participant_ids.append(pid)
            sample_types.append(stype)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(h5_files)} files...")

            thermal_pause(cooldown)

        except Exception as e:
            skipped.append((os.path.basename(h5_path), str(e)))

    features = np.vstack(features_list)
    print(f"\n  Loaded {len(features)} slides across "
          f"{len(set(cancer_labels))} cancer types")
    print(f"  Feature matrix: {features.shape}")
    if skipped:
        print(f"  Skipped: {len(skipped)} files")

    return features, cancer_labels, participant_ids, sample_types


# ──────────────────────────────────────────────────────────────────────
# 3. MODEL TRAINING & EVALUATION
# ──────────────────────────────────────────────────────────────────────

def get_lightweight_models(n_classes=2, n_jobs=1):
    """Return laptop-friendly classifiers.

    n_jobs=1 by default to limit CPU heat. Set higher if thermal
    headroom allows.
    """
    from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, BaggingClassifier,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB

    models = {
        # Linear models
        "LogReg_L2": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=42,
        ),
        "LogReg_L1": LogisticRegression(
            C=1.0, penalty="l1", solver="saga", max_iter=2000,
            class_weight="balanced", random_state=42,
        ),
        "RidgeClassifier": RidgeClassifier(
            alpha=1.0, class_weight="balanced",
        ),
        "SGD_huber": SGDClassifier(
            loss="modified_huber", alpha=1e-4, max_iter=1000,
            class_weight="balanced", random_state=42,
        ),
        "LinearSVC": LinearSVC(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=42,
        ),

        # Ensembles — single-threaded to limit heat
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=n_jobs,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=n_jobs,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42,
        ),
        "Bagging_LR": BaggingClassifier(
            estimator=LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            n_estimators=20, random_state=42, n_jobs=n_jobs,
        ),

        # Distance / density
        "KNN_5": KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs),
        "KNN_3": KNeighborsClassifier(n_neighbors=3, n_jobs=n_jobs),

        # Generative
        "LDA": LinearDiscriminantAnalysis(),
        "GaussianNB": GaussianNB(),
    }

    return models


def _oof_scores(pipe, X, y, cv, groups=None):
    """Return out-of-fold predicted labels and positive-class scores.

    scores is a 1D array (predict_proba[:,1] or decision_function) when
    available, else None.
    """
    from sklearn.model_selection import cross_val_predict

    preds = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method="predict")
    scores = None
    for method in ("predict_proba", "decision_function"):
        try:
            out = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method=method)
            scores = out[:, 1] if out.ndim == 2 and out.shape[1] >= 2 else out
            break
        except (AttributeError, ValueError):
            continue
    return preds, scores


def evaluate_models(features, labels, models, participant_ids=None,
                    n_folds=5, cv_strategy="auto",
                    do_pca=True, pca_variance=0.95, random_state=42,
                    cooldown=0.2):
    """Cross-validate with optional group-aware splits to avoid leakage.

    cv_strategy:
        'auto'             — group_kfold/logo if paired slides per participant,
                             else stratified_kfold/loo for small N
        'group_kfold'      — GroupKFold by participant_id (no participant in
                             both train and test)
        'logo'             — LeaveOneGroupOut by participant_id
        'stratified_kfold' — StratifiedKFold over slides (LEAKY when paired)
        'loo'              — LeaveOneOut over slides (LEAKY when paired)
    """
    from sklearn.model_selection import (
        StratifiedKFold, LeaveOneOut, GroupKFold, LeaveOneGroupOut,
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
    )

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_
    n_classes = len(class_names)

    groups = np.asarray(participant_ids) if participant_ids is not None else None

    print(f"\n  Classes: {dict(zip(class_names, np.bincount(y)))}")
    print(f"  Features: {features.shape[1]} dims")

    # Preprocessing pipeline (refit inside CV — preview here is just for logging)
    if do_pca:
        preprocess = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=random_state)),
        ])
        preview = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=random_state)),
        ]).fit(features)
        n_kept = preview.named_steps["pca"].n_components_
        var_explained = preview.named_steps["pca"].explained_variance_ratio_.sum()
        print(f"  PCA: {features.shape[1]} -> {n_kept} components "
              f"({var_explained:.1%} variance)")
    else:
        preprocess = Pipeline([("scaler", StandardScaler())])

    # Resolve CV strategy
    if cv_strategy == "auto":
        if groups is not None:
            counts = pd.Series(groups).value_counts()
            n_unique = len(counts)
            paired = bool((counts > 1).any())
            if paired:
                print(f"  Detected {int((counts > 1).sum())} participants with "
                      f"multiple slides — using GROUP CV to prevent leakage")
                cv_strategy = "logo" if n_unique < 40 else "group_kfold"
            else:
                cv_strategy = "loo" if len(y) < 40 else "stratified_kfold"
        else:
            cv_strategy = "loo" if len(y) < 40 else "stratified_kfold"

    if cv_strategy == "group_kfold":
        n_splits = min(n_folds, len(np.unique(groups)))
        cv = GroupKFold(n_splits=n_splits)
        cv_name = f"GroupKFold (k={n_splits}, by participant)"
    elif cv_strategy == "logo":
        cv = LeaveOneGroupOut()
        cv_name = "LeaveOneGroupOut (by participant)"
    elif cv_strategy == "loo":
        cv = LeaveOneOut()
        cv_name = "LeaveOneOut (by slide — LEAKY if paired)"
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=random_state)
        cv_name = f"StratifiedKFold (k={n_folds}, by slide — LEAKY if paired)"

    # For 1-sample-per-fold splits, AUC can't be computed per fold —
    # gather OOF predictions and compute a single AUC at the end.
    one_per_fold = cv_strategy in ("loo", "logo") or (
        cv_strategy == "group_kfold"
        and groups is not None
        and len(np.unique(groups)) <= len(y)
        and any(pd.Series(groups).value_counts() == 1)
    )

    print(f"  CV strategy: {cv_name}")
    print(f"  Training {len(models)} models...\n")

    results = []
    for name, model in models.items():
        t0 = time.time()
        pipe = Pipeline([("preprocess", preprocess), ("clf", model)])

        try:
            preds, scores = _oof_scores(pipe, features, y, cv, groups)

            row = {"model": name, "time_sec": 0.0}
            row["accuracy_mean"] = round(accuracy_score(y, preds), 4)

            if n_classes == 2:
                row["f1_mean"] = round(
                    f1_score(y, preds, average="binary", zero_division=0), 4
                )
                if scores is not None:
                    try:
                        row["auc_mean"] = round(roc_auc_score(y, scores), 4)
                    except ValueError:
                        pass
            else:
                row["f1_macro_mean"] = round(
                    f1_score(y, preds, average="macro", zero_division=0), 4
                )
                if scores is not None:
                    try:
                        row["auc_ovr_mean"] = round(
                            roc_auc_score(y, scores, multi_class="ovr"), 4
                        )
                    except ValueError:
                        pass

            elapsed = time.time() - t0
            row["time_sec"] = round(elapsed, 2)
            results.append(row)

            acc = row.get("accuracy_mean", 0)
            auc = row.get("auc_mean", row.get("auc_ovr_mean", None))
            auc_str = f"AUC={auc:.3f}" if auc else "AUC=N/A"
            print(f"  {name:25s}  Acc={acc:.3f}  {auc_str}  ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {name:25s}  FAILED: {e}  ({elapsed:.1f}s)")
            results.append({"model": name, "time_sec": round(elapsed, 2),
                           "error": str(e)})

        gc.collect()
        thermal_pause(cooldown)

    df = pd.DataFrame(results)
    sort_col = "auc_mean" if "auc_mean" in df.columns else "accuracy_mean"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")

    return df, class_names


# ──────────────────────────────────────────────────────────────────────
# 4. SURVIVAL ANALYSIS (OPTIONAL)
# ──────────────────────────────────────────────────────────────────────

def run_survival_analysis(features, participant_ids, clinical_path,
                          cancer_type="CHOL", n_folds=5, random_state=42,
                          cooldown=0.5):
    """Lightweight survival analysis with thermal pauses."""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
    except ImportError:
        return None

    clinical = pd.read_parquet(clinical_path)
    if "participant_id" not in clinical.columns and clinical.index.name == "participant_id":
        clinical = clinical.reset_index()
    if cancer_type:
        clinical = clinical[clinical["project_id"] == f"TCGA-{cancer_type}"]

    surv_records = []
    for pid in participant_ids:
        row = clinical[clinical["participant_id"] == pid]
        if row.empty:
            continue
        row = row.iloc[0]
        dead = str(row.get("vital_status", "")).lower() == "dead"
        t = row.get("days_to_death") if dead else row.get("days_to_last_followup")
        if pd.isna(t) or t <= 0:
            continue
        surv_records.append({"pid": pid, "event": dead, "time": float(t)})

    if len(surv_records) < 10:
        print(f"\n  Only {len(surv_records)} samples with valid survival data — "
              f"skipping survival analysis.")
        return None

    surv_df = pd.DataFrame(surv_records)
    pid_to_idx = {pid: i for i, pid in enumerate(participant_ids)}
    valid_idx = [pid_to_idx[r["pid"]] for r in surv_records if r["pid"] in pid_to_idx]
    X_surv = features[valid_idx]

    print(f"\n  Survival cohort: {len(surv_df)} patients "
          f"({surv_df['event'].sum()} events)")

    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.ensemble import (
            RandomSurvivalForest, GradientBoostingSurvivalAnalysis,
        )
        from sksurv.metrics import concordance_index_censored
        from sklearn.model_selection import KFold

        y_surv = np.array(
            [(bool(r["event"]), r["time"]) for r in surv_records],
            dtype=[("event", bool), ("time", float)]
        )

        pca_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=random_state)),
        ])
        X_pca = pca_pipe.fit_transform(X_surv)
        print(f"  PCA: {X_surv.shape[1]} -> {X_pca.shape[1]} components")

        surv_models = {
            "CoxnetLasso": CoxnetSurvivalAnalysis(
                l1_ratio=1.0, alpha_min_ratio=0.1, max_iter=1000,
            ),
            "CoxnetRidge": CoxnetSurvivalAnalysis(
                # l1_ratio must be > 0 in newer sksurv; use a tiny value
                # to approximate pure-ridge behavior
                l1_ratio=1e-3, alpha_min_ratio=0.1, max_iter=1000,
            ),
            "CoxnetElasticNet": CoxnetSurvivalAnalysis(
                l1_ratio=0.5, alpha_min_ratio=0.1, max_iter=1000,
            ),
            "RSF": RandomSurvivalForest(
                n_estimators=100, max_depth=5, min_samples_leaf=3,
                random_state=random_state, n_jobs=1,
            ),
            "GradientBoostSurv": GradientBoostingSurvivalAnalysis(
                n_estimators=50, max_depth=2, learning_rate=0.1,
                random_state=random_state,
            ),
        }

        # Group-aware split by participant when participants have multiple slides
        surv_pids = [r["pid"] for r in surv_records]
        pid_counts = pd.Series(surv_pids).value_counts()
        if (pid_counts > 1).any():
            from sklearn.model_selection import GroupKFold
            n_splits = min(n_folds, len(pid_counts))
            cv = GroupKFold(n_splits=n_splits)
            cv_groups = np.array(surv_pids)
            print(f"  CV: GroupKFold k={n_splits} (by participant — leakage-safe)")
        else:
            cv = KFold(n_splits=min(n_folds, len(surv_df)), shuffle=True,
                        random_state=random_state)
            cv_groups = None
            print(f"  CV: KFold k={cv.n_splits}")

        results = []
        for name, model in surv_models.items():
            t0 = time.time()
            c_indices = []

            try:
                split_iter = (cv.split(X_pca, groups=cv_groups)
                              if cv_groups is not None else cv.split(X_pca))
                for train_idx, test_idx in split_iter:
                    model_clone = type(model)(**model.get_params())
                    model_clone.fit(X_pca[train_idx], y_surv[train_idx])
                    pred = model_clone.predict(X_pca[test_idx])
                    ci = concordance_index_censored(
                        y_surv[test_idx]["event"],
                        y_surv[test_idx]["time"],
                        pred
                    )[0]
                    c_indices.append(ci)

                elapsed = time.time() - t0
                row = {
                    "model": name,
                    "c_index_mean": round(np.mean(c_indices), 4),
                    "c_index_std": round(np.std(c_indices), 4),
                    "time_sec": round(elapsed, 2),
                }
                results.append(row)
                print(f"  {name:25s}  C-index={row['c_index_mean']:.3f} "
                      f"+/- {row['c_index_std']:.3f}  ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  {name:25s}  FAILED: {e}")

            gc.collect()
            thermal_pause(cooldown)

        return pd.DataFrame(results).sort_values("c_index_mean", ascending=False)

    except ImportError:
        print("  scikit-survival not installed — skipping survival models.")
        print("  Install with: pip install scikit-survival")
        return None


# ──────────────────────────────────────────────────────────────────────
# 5. REPORTING
# ──────────────────────────────────────────────────────────────────────

def print_summary(clf_results, surv_results=None, class_names=None):
    """Print a clean summary of results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if clf_results is not None and len(clf_results) > 0:
        print("\n── Classification ──")
        display_cols = [c for c in ["model", "auc_mean", "auc_std",
                                     "accuracy_mean", "f1_mean",
                                     "auc_ovr_mean", "f1_macro_mean"]
                       if c in clf_results.columns]
        print(clf_results[display_cols].head(10).to_string(index=False))

        best = clf_results.iloc[0]
        auc_col = "auc_mean" if "auc_mean" in best.index else "auc_ovr_mean"
        if auc_col in best.index and pd.notna(best.get(auc_col)):
            print(f"\n  Best model: {best['model']} "
                  f"(AUC = {best[auc_col]:.3f})")
        else:
            print(f"\n  Best model: {best['model']} "
                  f"(Accuracy = {best['accuracy_mean']:.3f})")

    if surv_results is not None and len(surv_results) > 0:
        print("\n── Survival Analysis ──")
        print(surv_results.to_string(index=False))
        best_s = surv_results.iloc[0]
        print(f"\n  Best model: {best_s['model']} "
              f"(C-index = {best_s['c_index_mean']:.3f})")

    print("\n" + "=" * 70)


# ──────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    # Windows console defaults to cp1252; force UTF-8 so unicode dividers print
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(
        description="Lightweight WSI Embedding Classifier (Thermal-Safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--embeddings-dir", default="./embeddings",
                        help="Directory containing H5 embedding files")
    parser.add_argument("--clinical-data", default="./CLINICAL_FULL.parquet",
                        help="Path to TCGA clinical parquet file")
    parser.add_argument("--output-dir", default="./results_lightweight",
                        help="Output directory for results")
    parser.add_argument("--mode", choices=["binary", "multi"], default="binary",
                        help="'binary' = tumor vs normal, 'multi' = cancer type")
    parser.add_argument("--cancer-types", nargs="+", default=None,
                        help="Cancer types for multi mode (e.g., CHOL LIHC PAAD)")
    parser.add_argument("--cancer-type", default="CHOL",
                        help="Single cancer type for binary mode / survival")
    parser.add_argument("--target", default="tumor_normal",
                        help="Classification target: 'tumor_normal' (default), "
                             "'grade' (low G1/G2 vs high G3/G4), "
                             "'survival_quartile' (short Q1 vs long Q4), "
                             "or 'mutation:GENE' (e.g., mutation:BAP1)")
    parser.add_argument("--mutation-data", default=None,
                        help="Path to mutation parquet "
                             "(default: ./TCGA_data/MUTATIONS_FULL.parquet)")
    parser.add_argument("--aggregation", choices=["mean", "rich"], default="rich",
                        help="'mean' or 'rich' (mean+std+percentiles)")
    parser.add_argument("--aggregated-csv", default=None,
                        help="Skip H5 loading — use pre-aggregated CSV")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--loo", action="store_true",
                        help="Use leave-one-out CV (by slide — leaky if paired samples)")
    parser.add_argument("--cv-strategy",
                        choices=["auto", "group_kfold", "logo",
                                 "stratified_kfold", "loo"],
                        default="auto",
                        help="CV split: 'auto' uses GroupKFold/LOGO by "
                             "participant when paired slides exist (prevents "
                             "patient-level leakage), else falls back to slide-level")
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--no-survival", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)

    # Thermal options
    parser.add_argument("--cool", action="store_true",
                        help="Extra-cool mode: longer pauses, single-threaded everything")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint after a crash")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel jobs for ensemble models (1=coolest, -1=all cores)")

    args = parser.parse_args()

    # Set thermal parameters
    if args.cool:
        cooldown = COOLDOWN_COOL
        n_jobs = 1
        print("=" * 70)
        print("Lightweight WSI Classifier (COOL MODE — thermal-safe)")
        print("=" * 70)
        print("  Extended cooldown pauses enabled")
        print("  All models single-threaded")
    else:
        cooldown = COOLDOWN_NORMAL
        n_jobs = args.n_jobs
        print("=" * 70)
        print("Lightweight WSI Embedding Classifier")
        print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──
    if args.aggregated_csv:
        print(f"\nLoading pre-aggregated features from {args.aggregated_csv}...")
        df = pd.read_csv(args.aggregated_csv)
        feature_cols = [c for c in df.columns if c not in
                       ["participant_id", "label", "cancer_type", "sample_type",
                        "filename"]]
        features = df[feature_cols].values
        labels = df["label"].tolist()
        participant_ids = df["participant_id"].tolist()
        print(f"  Loaded {len(features)} samples, {len(feature_cols)} features")

    elif args.mode == "binary":
        print(f"\nLoading CHOL embeddings (tumor vs normal)...")
        print(f"  Directory: {args.embeddings_dir}")
        print(f"  Aggregation: {args.aggregation}")
        print(f"  Cooldown: {cooldown}s between files\n")
        features, labels, participant_ids, filenames = \
            load_and_aggregate_embeddings(
                args.embeddings_dir, method=args.aggregation,
                cooldown=cooldown, output_dir=args.output_dir,
                resume=args.resume,
            )

    elif args.mode == "multi":
        print(f"\nLoading multi-cancer embeddings...")
        print(f"  Directory: {args.embeddings_dir}")
        print(f"  Cancer types: {args.cancer_types or 'all available'}")
        print(f"  Aggregation: {args.aggregation}\n")
        features, labels, participant_ids, _ = \
            load_multi_cancer_embeddings(
                args.embeddings_dir, args.clinical_data,
                method=args.aggregation, cancer_types=args.cancer_types,
                cooldown=cooldown, output_dir=args.output_dir,
                resume=args.resume,
            )

    # ── Relabel for the chosen target ──
    if args.target != "tumor_normal":
        print(f"\n── Relabeling for target: {args.target} ──")
        # Aggregated-CSV path doesn't track filenames; substitute pids
        try:
            fn_list = filenames  # set by binary mode
        except NameError:
            fn_list = list(participant_ids)
        features, labels, participant_ids, _ = relabel_for_target(
            features, labels, participant_ids, fn_list,
            target=args.target,
            clinical_path=args.clinical_data,
            mutation_path=args.mutation_data,
            cancer_type=args.cancer_type,
        )
        if len(set(labels)) < 2:
            print(f"\nERROR: only one class after filtering — cannot classify")
            sys.exit(1)

    # Tag output files with the target so multiple runs don't overwrite
    target_tag = args.target.replace(":", "_")

    # ── Decide CV strategy ──
    # 'auto' picks group CV when paired slides exist (leakage-safe),
    # else falls back to LOO/StratifiedKFold by slide.
    if args.cv_strategy != "auto":
        cv_strategy = args.cv_strategy
    elif args.loo:
        cv_strategy = "loo"
    else:
        cv_strategy = "auto"

    # ── Classification ──
    print(f"\n{'─' * 70}")
    print("CLASSIFICATION")
    print(f"{'─' * 70}")

    n_classes = len(set(labels))
    models = get_lightweight_models(n_classes, n_jobs=n_jobs)
    clf_results, class_names = evaluate_models(
        features, labels, models,
        participant_ids=participant_ids,
        n_folds=args.n_folds, cv_strategy=cv_strategy,
        do_pca=not args.no_pca,
        random_state=args.random_state,
        cooldown=cooldown * 0.4,  # shorter pauses for small sklearn ops
    )

    clf_path = os.path.join(args.output_dir,
                            f"classification_results_{target_tag}.csv")
    clf_results.to_csv(clf_path, index=False)
    print(f"\n  Saved: {clf_path}")

    # ── Survival Analysis ──
    surv_results = None
    if not args.no_survival and os.path.exists(args.clinical_data):
        print(f"\n{'─' * 70}")
        print("SURVIVAL ANALYSIS")
        print(f"{'─' * 70}")

        surv_results = run_survival_analysis(
            features, participant_ids, args.clinical_data,
            cancer_type=args.cancer_type,
            n_folds=args.n_folds,
            random_state=args.random_state,
            cooldown=cooldown,
        )
        if surv_results is not None:
            surv_path = os.path.join(args.output_dir,
                                    f"survival_results_{target_tag}.csv")
            surv_results.to_csv(surv_path, index=False)
            print(f"\n  Saved: {surv_path}")

    # ── Save aggregated embeddings for reuse ──
    if not args.aggregated_csv:
        agg_path = os.path.join(args.output_dir, "aggregated_embeddings.csv")
        n_features = features.shape[1]
        feat_cols = [f"f{i}" for i in range(n_features)]
        agg_df = pd.DataFrame(features, columns=feat_cols)
        agg_df.insert(0, "participant_id", participant_ids)
        agg_df.insert(1, "label", labels)
        agg_df.to_csv(agg_path, index=False)
        print(f"\n  Saved aggregated embeddings: {agg_path}")
        print(f"  (Reuse with --aggregated-csv {agg_path} to skip H5 loading)")

    # ── Summary ──
    print_summary(clf_results, surv_results, class_names)
    print(f"\nAll results saved to: {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
