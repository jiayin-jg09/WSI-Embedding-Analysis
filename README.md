# WSI Embedding Analysis · TCGA Pan-Cancer

A consolidated, leakage-safe, thermal-safe pipeline that runs classification
and per-cancer survival analysis on pre-extracted whole-slide image
embeddings.

**Survival CV:** pooled leave-one-participant-out (LOPO) with 1000-iter
bootstrap 95% CI on the C-index, run per cancer type. Replaces the earlier
5-fold scheme that produced unreliable estimates (e.g. RSF 0.71 ± 0.22, where
the wide std reflected fold-split noise on n=55 rather than predictive power).

📖 **Full writeup with figures:** [project website](https://YOUR-USERNAME.github.io/YOUR-REPO/)
(replace once GitHub Pages is enabled — see [Deploying the website](#deploying-the-website))

---

## Quick start

```bash
pip install numpy pandas scipy scikit-learn h5py matplotlib seaborn
pip install scikit-survival lifelines  # for survival models

# Reproduce on the small CHOL cohort (uses cached aggregated CSV)
python wsi_survival_pipeline.py \
    --aggregated-csv results_lightweight/aggregated_embeddings.csv \
    --output-dir results_v2 --no-survival --target grade
```

Full run on H5 embeddings (auto-detects both `./embeddings` and
`./TCGA UNI2 embeddings`):

```bash
python wsi_survival_pipeline.py \
    --clinical-data ./CLINICAL_FULL.parquet \
    --output-dir ./results_v2
```

After a crash, resume aggregation:

```bash
python wsi_survival_pipeline.py --resume --output-dir ./results_v2
```

## What's in here

| File | Purpose |
|------|---------|
| `wsi_survival_pipeline.py` | Single consolidated pipeline. Classification + per-cancer pooled-LOPO survival with bootstrap CIs. Thermal-safe (chunked H5, cooldowns, checkpoint/resume). |
| `pooled_survival_analysis.py` | Pan-cancer **pooled** survival: one model across all 8 cohorts (cancer-stratified 5-fold CV), reporting overall *and* within-cancer-stratified C-index. Runs from the cached aggregated CSV + clinical parquet. |
| `generate_workflow_figures.py` | Generates the four pipeline diagrams in `figures/`. |
| `generate_model_figures.py` | Generates the illustrative example plots in `figures/models/` for the model-explainer pages (synthetic data, no cohort data). |
| `aggregate_full_cohort.py` | Thermal-safe re-aggregation of all 8-cancer slides from the raw H5 patch files, computing the full menu of **31 aggregation methods** (median, geometric mean, IQR, entropy, …) in a single pass per slide. One H5 at a time, cooldowns, checkpoint/resume every 50 slides. Output → `results_v2/agg_full/full_methods.npz` (+ `meta.csv`). Run: `python aggregate_full_cohort.py --resume --cooldown 2.0`. |
| `aggregation_method_comparison.py` | Ranks all 31 aggregation methods by **downstream prediction** using the cached `full_methods.npz` (no H5 reads): pan-cancer pooled survival (cancer-stratified 5-fold CoxnetLasso) + tumor-grade classification. Thermal-light first pass (fixed 50-PC PCA, 200-sample bootstrap, cooldown + per-method checkpoint). Output → `results_v2/agg_full/{survival,classification_grade}_method_comparison.csv`. Finding: central-tendency/percentile stats win; pure spread/shape stats lag. |
| `uni2_omics_correlation.py` | Correlates each of the 1,536 UNI2 embedding dimensions against molecular features (immune fractions/signatures, RPPA, mutations, expression, CNV) from the lab's TCGA multi-omics modules, joined on `participant_id`. Vectorized rank-correlation + BH-FDR, reported both **pooled** and **cancer-residualized** (within-cancer). UNI2 analogue of the mentor's CTransPath `dim_omic_correlations`. Output → `results_v2/dim_omics/uni2_<agg>_<omic>.parquet` (gitignored). Finding: UNI2 dims strongly encode the tumor microenvironment / expression state but barely encode specific mutations within cancer type. |
| `uni2_predictability.py` | Multivariate dimension exploration: how much of each molecular target (immune signatures, tumor purity, genomic-instability scores, driver mutations) the **whole** UNI2 embedding predicts, via PCA + RidgeCV / logistic regression, cancer-stratified 5-fold, reported pooled and within-cancer. Runs the **mean** block and the **std** block (intra-slide heterogeneity). Output `results_v2/dim_omics/predictability.csv` + `figures/uni2_predictability.png`. Finding: full-embedding within-cancer R² ≈ 0.24–0.29 for immune/instability/purity (far above any single dim); mutations stay ~0.59 AUC; the std block does not beat the mean block. |
| `uni2_tile_galleries.py` | For each top distinct dimension, takes the top 10 cases by the dimension's slide-level value and crops each case's 100 most-activating H&E tiles from its WSI (`.svs` via openslide, using the patch coords in the H5), assembling one montage per dimension (ten 10×10 per-case blocks) — "what each dimension looks at". Keyed by the catalog code (feature hidden). Reads WSIs from `--wsi-dir` (external drive, never copied); montages → `figures/galleries/`; builds the **Tiles** page (`pages/models-tiles.html`). |
| `uni2_catalog.py` | Builds a coded **discovery catalog** of the strongest within-cancer dimension↔feature links (top ~200 per omic), each with a stable opaque code (e.g. `U2-D0953-EXPR-7A3F2`), a **cross-model robustness** count (in how many of the 6 lab foundation models — ctranspath, kaiko-s8, lunit-dino, pathdino, sp22m, uni2 — the feature is top-20% encoded), and a **survival** HR (stratified Cox on OS); plus hub-dimension grouping and a separate mutation section. Writes `results_v2/dim_omics/catalog*.csv` (gitignored — the only place feature names live) and injects the table into the **Catalog** page. |
| `uni2_dimension_enrichment.py` | Internal/optional tool: annotates dimensions by gene-set enrichment of their top within-cancer-correlated genes. Local outputs only (not shown on the site); needs a local gene-set file. |
| `uni2_vs_ctranspath.py` | Phase 3 of the dim-omics analysis: compares UNI2 vs the mentor's **CTransPath** foundation model at the feature level (per-feature best \|rho\| over dims), asking whether two independently-trained models encode the same biology. Writes `results_v2/dim_omics/compare_*.csv` + `figures/uni2_vs_ctranspath.png` (agreement scatter grid) and `figures/uni2_biology_encoded.png` (pooled vs within-cancer encodability per omic). Finding: agreement Spearman 0.75-0.95 across omics. |
| `generate_pca_plots.py` | Generates real PCA scatter plots from the aggregated embeddings (`figures/pca/`): 8-cancer cohort by cancer type for the 6 cached rich methods **and** the full 31-method set re-aggregated by `aggregate_full_cohort.py` (median headline + all-methods grid), the full method sweep on CHOL by tumor/normal, plus clustering + FDR-significance figures (`figures/models/`). The 31-method by-cancer view needs `results_v2/agg_full/full_methods.npz`; the rest need no H5 files. |
| `index.html` + `pages/*.html` | Multi-page project website (Pine dark theme, dark/light toggle). `index.html` is the entry point at the repo root; the other pages (background, overview/story, methods, results, figures) live in `pages/`. Readable directly or served via GitHub Pages. |
| `pages/models-*.html` | "Models explained" pages (reached via the **Models** nav dropdown): survival, classification and aggregation models in depth, clustering & statistics, evaluation metrics, and **biology** (what molecular signal each embedding dimension encodes, plus the UNI2 vs CTransPath comparison). |
| `assets/styles.css`, `assets/app.js` | Shared theme + interactivity (theme toggle, nav dropdown, image lightbox, sortable tables, Chart.js charts, animated counters). |
| `METHODS_DETAILED_EXPLANATION.txt` | In-depth method notes for aggregation + modeling. |

## Prediction targets

```bash
--target tumor_normal       # default — AUC 1.00, saturated, not useful
--target grade              # low (G1/G2) vs high (G3/G4) — AUC 0.80, p = 0.005
--target mutation:GENE      # binary mut vs wt, e.g. mutation:BAP1
--target survival_quartile  # short (≤Q1) vs long (≥Q3)
```

## Cross-validation

**Classification** auto-picks a leakage-safe split: when any participant has
multiple slides (e.g. tumor + matched normal), `GroupKFold` /
`LeaveOneGroupOut` by `participant_id` is used so no participant appears in
both train and test.

**Survival** uses pooled leave-one-participant-out:

1. For each participant, fit the model on the other N−1 and predict their risk.
2. Pool all N held-out risk scores and compute a single C-index.
3. Bootstrap resample (event, time, risk) triples 1000× → 2.5/97.5 percentiles
   give a 95% CI.
4. Per (cancer, model), also report: hazard ratio per 1 SD of risk score
   (univariate lifelines Cox, with 95% CI and p), and log-rank chi² + p for
   the median-split risk groups.

This is the right choice when n is small (≤ a few hundred) and 5-fold leaves
some folds with too few events to estimate a meaningful per-fold C-index.

Every cancer is evaluated with three feature modes side-by-side so you can
read off the marginal value of imaging:

- `Cox_AgeSex` — clinical-only baseline (age + gender)
- `Cox_WSI_plus_Clin` — combined model (WSI PCs ⊕ clinical)
- `CoxnetLasso` / `CoxnetElasticNet` / `RSF` / `GradientBoostSurv` — WSI-only

> **Clinical-baseline caveat:** `tumor_stage` is NaN throughout
> `CLINICAL_FULL.parquet`, so the clinical baseline is `age + gender` only.
> A stronger clinical baseline (with stage) would set a higher bar for WSI;
> the current comparison is therefore *conservative*.

## Thermal-safe defaults

Built for laptops that overheat:

- H5 files loaded one at a time with explicit cooldown pauses
- `n_jobs = 1` by default for all parallelisable models
- Checkpointing every 5 files (recover with `--resume`)
- `--cool` mode for extended pauses

## Repository layout

```
JiaYin_WSI_Analysis/
├── wsi_survival_pipeline.py        # consolidated pipeline
├── generate_workflow_figures.py
├── index.html                      # project website entry point (home)
├── pages/                          # all other site pages (background, methods, results, models-*, …)
├── assets/                         # shared styles.css + app.js
├── figures/                        # pipeline diagrams
├── results_lightweight/            # historical results (small CHOL cohort)
├── results/                        # historical results (original 16-model)
├── requirements.txt
└── README.md
```

## Data (not in the repo)

These files exceed GitHub's size limits and are listed in `.gitignore`:

- `embeddings/*.h5` — UNI2 patch embeddings for the 59 TCGA-CHOL slides
- `TCGA UNI2 embeddings/*.h5` — 2075 slides across 8 TCGA cancer types
- `CLINICAL_FULL.parquet` — TCGA clinical metadata
- `TCGA_data/*.parquet` — multi-omics (expression, mutations, CNV, RPPA)

## Deploying the website

1. Push this repo to GitHub.
2. In repo Settings → Pages, set the source to the `main` branch, root folder.
3. The site is served at `https://<your-username>.github.io/<repo-name>/`.

The site is plain HTML/CSS/JS with no build step — pages share `assets/styles.css`
and `assets/app.js`, and Chart.js is loaded from a CDN. Jekyll is not required;
an optional `.nojekyll` file is included so `assets/` is served verbatim.

## Reproducing the permutation test

```bash
# Already produces results_lightweight/permutation_grade_aucs.csv and the PNG
# See the inline script in the run log for the exact 200-shuffle procedure.
```

The empirical p-value uses the Phipson & Smyth (2010) correction:
`p = (n_ge + 1) / (N + 1)` where `n_ge` counts shuffled AUCs ≥ baseline.
