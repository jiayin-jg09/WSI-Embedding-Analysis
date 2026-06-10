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
| `generate_workflow_figures.py` | Generates the four pipeline diagrams in `figures/`. |
| `index.html` + `background/overview/methods/results/figures.html` | Multi-page project website (Pine dark theme, dark/light toggle) — readable directly or served via GitHub Pages. |
| `assets/styles.css`, `assets/app.js` | Shared theme + interactivity (theme toggle, image lightbox, sortable tables, Chart.js charts, animated counters). |
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
├── index.html                      # project website (home)
├── background.html overview.html methods.html results.html figures.html
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
