# WSI Embedding Analysis Pipeline

Comprehensive analysis pipeline for Whole Slide Image (WSI) embeddings using UNI2 foundation model features. Benchmarks **43 aggregation methods** across **24 classification models**, **16 survival models**, and clustering analysis.

**Target dataset**: TCGA-CHOL (Cholangiocarcinoma) - 39 tumor + 20 normal = 59 slides

**Status**: Ready to run. No analyses have been executed yet -- all results will be generated on first run.

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the analysis
bash run.sh            # Linux/Mac
run.bat                # Windows
```

Or run directly:

```bash
python wsi_embedding_analysis.py \
    --embeddings-dir ./embeddings \
    --clinical-data ./CLINICAL_FULL.parquet \
    --output-dir ./results \
    --cancer-type CHOL
```

## System Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.8+ |
| RAM | 16 GB (32 GB recommended) |
| Disk | ~500 MB for results (embeddings require additional space) |
| GPU | Not required (CPU-only pipeline) |

## Directory Structure

```
JiaYin_WSI_Analysis/
├── wsi_embedding_analysis.py       # Main analysis script
├── requirements.txt                # Python dependencies
├── run.sh / run.bat                # Launchers (relative paths, auto-venv)
├── setup_data.sh                   # Data copy helper (for rebuilding package)
├── generate_workflow_figures.py     # Pipeline diagram generator
├── METHODS_DETAILED_EXPLANATION.txt # Detailed guide to all 43 aggregation + 40 modeling methods
├── CLINICAL_FULL.parquet           # TCGA clinical data (all cancer types)
├── embeddings/                     # H5 patch-level embeddings from UNI2
│   ├── TCGA-W5-AA2R-01Z-00-DX1.*.h5   (39 tumor slides, sample code 01)
│   ├── TCGA-W5-AA2Q-11A-01-TSA.*.h5   (20 normal slides, sample code 11)
│   └── ... (59 total)
├── TCGA_data/                      # Supplementary TCGA multi-omics data
│   ├── EXPRESSION_FULL.parquet     # RNA-seq TPM (60,498 genes, ~890 MB)
│   ├── MUTATIONS_FULL.parquet      # Somatic mutations binary matrix (~19 MB)
│   ├── CNV_FULL.parquet            # Copy number variation (~549 MB)
│   └── RPPA_FULL.parquet           # Protein expression (~9 MB)
├── figures/                        # Pipeline workflow diagrams
│   ├── fig1_pipeline_overview.png
│   ├── fig2_aggregation_methods.png
│   ├── fig3_modeling_pipeline.png
│   └── fig4_results_outputs.png
└── results/                        # Created by the pipeline on first run
    ├── aggregated_embeddings/      # 43 CSV files (one per method)
    ├── classification_results/     # AUC, F1, accuracy per model
    ├── survival_results/           # C-index per model
    ├── clustering_results/         # K-means, hierarchical results
    ├── statistical_tests/          # FDR-corrected p-values
    ├── figures/                    # Heatmaps, PCA plots
    └── summary_report.csv          # Best results per aggregation method
```

## What the Pipeline Does

### 1. Patch Aggregation (43 methods)

Converts variable-length patch embeddings (N patches x 1536 dims) into fixed-size slide-level representations:

| Category | Methods | Output Dim |
|----------|---------|------------|
| Central tendency | mean, median, trimmed_mean, winsorized_mean, geometric_mean, harmonic_mean | 1,536 |
| Dispersion | std, variance, range, CV, IQR, MAD, mid_range | 1,536 |
| Distribution shape | entropy, skewness, kurtosis | 1,536 |
| Percentiles | P5, P10, P25, P50, P75, P90, P95 | 1,536 |
| Extreme values | max, min, top5_mean, top10_mean, bottom5/10_mean, top5_L1, median_patch | 1,536 |
| Weighted | entropy_weighted_mean, variance_weighted_mean | 1,536 |
| Energy | RMS, sum_absolute_values | 1,536 |
| Mega-vectors | Concatenated multi-stat vectors | 4,608 - 10,752 |

### 2. Classification (24 models)

Tumor vs normal tissue classification with stratified 5-fold and leave-one-out CV:

- **Tree ensembles**: RandomForest, ExtraTrees, GradientBoosting, HistGB, AdaBoost, Bagging, DecisionTree
- **Boosting** (optional): XGBoost, LightGBM, CatBoost
- **Linear**: LogisticRegression (L1/L2/ElasticNet), Ridge, SGD
- **SVM**: LinearSVC, SVC-linear, SVC-RBF, NuSVC
- **Other**: KNN, NearestCentroid, GaussianNB, BernoulliNB, LDA

Metrics: AUC-ROC, F1, accuracy, precision, recall

### 3. Survival Analysis (16 models)

Prognostic modeling with C-index evaluation:

- **scikit-survival**: CoxPH, Coxnet (Lasso/Ridge/ElasticNet), GradientBoosting, ComponentwiseGB, RandomSurvivalForest, ExtraSurvivalTrees, FastSurvivalSVM, FastKernelSurvivalSVM, IPCRidge
- **lifelines**: CoxPHFitter, WeibullAFT, LogNormalAFT, LogLogisticAFT

### 4. Clustering & Statistical Tests

- PCA (95% variance threshold) + K-means + Ward hierarchical clustering
- Chi-square test (cluster labels vs tumor/normal)
- Log-rank test (cluster labels vs survival outcomes)
- Benjamini-Hochberg FDR correction

## Input Data Format

### H5 Embedding Files

Each `.h5` file contains patch-level UNI2 embeddings for one WSI slide:

- **Key**: `features`, `embeddings`, `feats`, or `data`
- **Shape**: `(N_patches, 1536)` where N varies per slide
- **Filename**: Must contain TCGA barcode (e.g., `TCGA-W5-AA2R-01A-01-TS1.h5`)
  - Characters 1-12: participant ID (e.g., `TCGA-W5-AA2R`)
  - Characters 14-15: sample type (`01`-`09` = tumor, `10`-`19` = normal)

### Clinical Data (Parquet)

TCGA clinical data with columns:
- `project_id`: e.g., `TCGA-CHOL` (for filtering by cancer type)
- `days_to_death`, `days_to_last_followup`: survival times
- `vital_status`: `Dead` / `Alive`

## Command-Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--embeddings-dir` | `./embeddings` | Directory with H5 files |
| `--clinical-data` | `./CLINICAL_FULL.parquet` | Path to clinical parquet |
| `--output-dir` | `./results` | Output directory |
| `--cancer-type` | `CHOL` | TCGA cancer type code |
| `--n-folds` | `5` | Cross-validation folds |
| `--random-state` | `42` | Random seed |

## Optional Dependencies

The pipeline degrades gracefully if optional packages are missing:

| Package | What's lost without it |
|---------|----------------------|
| `scikit-survival` | 11 survival models (sksurv-based) skipped |
| `lifelines` | 4 survival models (AFT-based) + log-rank tests skipped |
| `xgboost` | XGBClassifier skipped |
| `lightgbm` | LGBMClassifier skipped |
| `catboost` | CatBoostClassifier skipped |

Install only the required packages for a minimal run:

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn h5py
```

## Supplementary TCGA Data

The `TCGA_data/` folder contains additional multi-omics data for all 33 TCGA cancer types (not just CHOL). These files are **not used by the main pipeline** but are provided for potential downstream multi-omics analyses:

| File | Contents | Size |
|------|----------|------|
| `EXPRESSION_FULL.parquet` | RNA-seq TPM values for 60,498 genes across 11,428 samples | 890 MB |
| `MUTATIONS_FULL.parquet` | Binary somatic mutation status for 21,306 genes | 19 MB |
| `CNV_FULL.parquet` | Gene-level copy number variation for 24,776 genes | 549 MB |
| `RPPA_FULL.parquet` | Protein expression (RPPA) for 640 proteins | 9 MB |

All files are indexed by `participant_id` and can be filtered to CHOL with:
```python
df = pd.read_parquet("TCGA_data/EXPRESSION_FULL.parquet")
# Filter to CHOL participants using clinical data
clinical = pd.read_parquet("CLINICAL_FULL.parquet")
chol_ids = clinical[clinical['project_id'] == 'TCGA-CHOL']['participant_id']
df_chol = df[df.index.isin(chol_ids)]
```

## Additional Documentation

- **`METHODS_DETAILED_EXPLANATION.txt`** - In-depth explanation of all 43 aggregation methods and 40 modeling methods, including formulas, rationale, interpretation, and summary tables.

## Generating Workflow Figures

```bash
python generate_workflow_figures.py
```

This creates 4 pipeline diagrams in `figures/` (pre-generated figures are already included):
1. **Pipeline Overview** - End-to-end WSI analysis flow
2. **Aggregation Methods** - Taxonomy of all 43 methods
3. **Modeling Pipeline** - Classification, survival, and clustering branches
4. **Results & Outputs** - Output file structure and interpretation
