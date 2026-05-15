# WSI Embedding Analysis · TCGA-CHOL

A lightweight, leakage-safe, thermal-safe pipeline that explores what
pre-extracted whole-slide image embeddings can actually predict on a small
clinical cohort.

**Headline result:** tumor grade (low G1/G2 vs high G3/G4) classification on
TCGA cholangiocarcinoma WSIs reaches AUC = 0.80 (KNN-5, leave-one-out CV,
N = 26, perfectly balanced), validated by a 200-shuffle permutation test
(p = 0.005).

📖 **Full writeup with figures:** [project website](https://YOUR-USERNAME.github.io/YOUR-REPO/)
(replace once GitHub Pages is enabled — see [Deploying the website](#deploying-the-website))

---

## Quick start

```bash
pip install numpy pandas scipy scikit-learn h5py matplotlib seaborn
pip install scikit-survival lifelines  # optional, for survival models

# Reproduce the validated grade result (uses the cached aggregated CSV)
python lightweight_wsi_classifier.py \
    --aggregated-csv results_lightweight/aggregated_embeddings.csv \
    --target grade --output-dir results_lightweight --no-survival
```

If you have the H5 embeddings, run from scratch:

```bash
python lightweight_wsi_classifier.py \
    --embeddings-dir ./embeddings \
    --clinical-data ./CLINICAL_FULL.parquet
```

## What's in here

| File | Purpose |
|------|---------|
| `lightweight_wsi_classifier.py` | Main thermal-safe pipeline. 13 classifiers, group-aware CV, four prediction targets, permutation-friendly. |
| `wsi_embedding_analysis.py` | Comprehensive 43-aggregation × 24-classifier × 16-survival benchmark from the original project. |
| `generate_workflow_figures.py` | Generates the four pipeline diagrams in `figures/`. |
| `index.html` | Project website (Pine dark theme) — readable directly or served via GitHub Pages. |
| `METHODS_DETAILED_EXPLANATION.txt` | In-depth method notes for all 43 aggregations + 40 modeling methods. |

## Prediction targets

```bash
--target tumor_normal       # default — AUC 1.00, saturated, not useful
--target grade              # low (G1/G2) vs high (G3/G4) — AUC 0.80, p = 0.005
--target mutation:GENE      # binary mut vs wt, e.g. mutation:BAP1
--target survival_quartile  # short (≤Q1) vs long (≥Q3)
```

## Cross-validation

The pipeline auto-selects a leakage-safe CV strategy. When matched
tumor/normal slides exist for the same participant (as they do in TCGA-CHOL —
15 of 39 participants), `GroupKFold` / `LeaveOneGroupOut` by `participant_id`
is used so no participant appears in both train and test.

```bash
--cv-strategy auto              # leakage-safe by default (recommended)
--cv-strategy group_kfold       # explicit GroupKFold by participant
--cv-strategy logo              # LeaveOneGroupOut by participant
--cv-strategy stratified_kfold  # slide-level (leaky if paired)
--cv-strategy loo               # slide-level LOO (leaky if paired)
```

## Thermal-safe defaults

Built for laptops that overheat:

- H5 files loaded one at a time with explicit cooldown pauses
- `n_jobs = 1` by default for all parallelisable models
- Checkpointing every 5 files (recover with `--resume`)
- `--cool` mode for extended pauses

## Repository layout

```
JiaYin_WSI_Analysis/
├── lightweight_wsi_classifier.py   # thermal-safe pipeline
├── wsi_embedding_analysis.py       # 43-aggregation benchmark
├── generate_workflow_figures.py
├── index.html                      # project website
├── figures/                        # pipeline diagrams
├── results_lightweight/            # CSVs + permutation null PNG
├── requirements.txt
└── README.md
```

## Data (not in the repo)

These files exceed GitHub's size limits and are listed in `.gitignore`:

- `embeddings/*.h5` — UNI2 patch embeddings for the 59 TCGA-CHOL slides
- `CLINICAL_FULL.parquet` — TCGA clinical metadata
- `TCGA_data/*.parquet` — multi-omics (expression, mutations, CNV, RPPA)

The pipeline regenerates everything in `results_lightweight/` given those inputs.

## Deploying the website

1. Push this repo to GitHub.
2. In repo Settings → Pages, set the source to the `main` branch, root folder.
3. The site is served at `https://<your-username>.github.io/<repo-name>/`.

The `index.html` is self-contained (inline CSS, no build step). Jekyll is not
required; an optional `.nojekyll` file is included to skip Jekyll processing.

## Reproducing the permutation test

```bash
# Already produces results_lightweight/permutation_grade_aucs.csv and the PNG
# See the inline script in the run log for the exact 200-shuffle procedure.
```

The empirical p-value uses the Phipson & Smyth (2010) correction:
`p = (n_ge + 1) / (N + 1)` where `n_ge` counts shuffled AUCs ≥ baseline.
