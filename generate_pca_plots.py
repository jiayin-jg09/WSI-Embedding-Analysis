"""
generate_pca_plots.py
=====================
Real PCA scatter plots of the aggregated WSI embeddings, plus clustering /
statistics figures, for the website's explainer pages. All from CACHED data
(no 93 GB H5 files):

  Part 1 — 8-cancer cohort, colored by cancer type, for the 6 cached "rich"
           aggregation blocks (mean, std, P10, P25, P75, P90).
  Part 2 — CHOL-59 cohort, full ~44-method sweep, colored by tumor vs normal.
  Part 3 — clustering (silhouette / cluster scatter) + FDR significance.

Outputs: figures/pca/*.png, figures/pca/chol/*.png, figures/models/clustering_*.png,
figures/models/fdr_significance.png

Run:  python generate_pca_plots.py
"""
import os, json, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

AGG_CSV = "results_v2/aggregated_embeddings.csv"
CLINICAL = "CLINICAL_FULL.parquet"
CHOL_DIR = "results/aggregated_embeddings"
CLUST_JSON = "results/clustering_results/all_results.json"
STATS_CSV = "results/statistical_tests/fdr_corrected_results.csv"

PCA_DIR = os.path.join("figures", "pca")
CHOL_OUT = os.path.join(PCA_DIR, "chol")
MODELS_DIR = os.path.join("figures", "models")
for d in (PCA_DIR, CHOL_OUT, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

# rich-block layout (verified): each block = 1536 dims, in this order
RICH_BLOCKS = [("mean", 0), ("std", 1), ("P10", 2), ("P25", 3), ("P75", 4), ("P90", 5)]
BLOCK = 1536
CANCER_COLORS = {
    "TCGA-COAD": "#0d9488", "TCGA-STAD": "#2dd4bf", "TCGA-LIHC": "#5eead4",
    "TCGA-CESC": "#38bdf8", "TCGA-ESCA": "#6366f1", "TCGA-READ": "#a855f7",
    "TCGA-ACC": "#f59e0b", "TCGA-CHOL": "#ef4444",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#b9c8c1", "axes.labelcolor": "#16241d",
    "text.color": "#16241d", "xtick.color": "#5a6f66", "ytick.color": "#5a6f66",
    "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
    "axes.grid": True, "grid.color": "#e3ebe7", "grid.linewidth": 0.8,
})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def pca2(X):
    Xs = StandardScaler().fit_transform(X)
    return PCA(n_components=2, random_state=0).fit_transform(Xs)


# ============================================================ Part 1
def part1_bycancer():
    print("Part 1: 8-cancer PCA by cancer type ...")
    from wsi_survival_pipeline import load_aggregated_csv
    feats, sample_types, pids, _ = load_aggregated_csv(AGG_CSV)
    feats = np.asarray(feats); sample_types = np.asarray(sample_types)
    pids = np.asarray(pids)
    tumor = sample_types == "tumor"
    feats, pids = feats[tumor], pids[tumor]

    clin = pd.read_parquet(CLINICAL)
    if "participant_id" not in clin.columns:
        clin.index.name = "participant_id"; clin = clin.reset_index()
    proj = clin.drop_duplicates("participant_id").set_index("participant_id")["project_id"]
    cancer = np.array([proj.get(p, None) for p in pids], dtype=object)
    keep = np.array([c in CANCER_COLORS for c in cancer])
    feats, cancer = feats[keep], cancer[keep]
    print(f"  {len(feats)} tumor slides with known cancer type")

    def scatter(ax, P, title):
        for c, col in CANCER_COLORS.items():
            m = cancer == c
            if m.any():
                ax.scatter(P[m, 0], P[m, 1], s=9, c=col, alpha=0.6,
                           edgecolor="none", label=c.replace("TCGA-", ""))
        ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_xticks([]); ax.set_yticks([])

    # headline: mean
    Pmean = pca2(feats[:, 0:BLOCK])
    fig, ax = plt.subplots(figsize=(6.6, 5))
    scatter(ax, Pmean, "PCA of slide embeddings (mean aggregation)\ncolored by cancer type")
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.9, markerscale=1.6)
    save(fig, os.path.join(PCA_DIR, "pca_mean_bycancer.png"))

    # 6-panel grid
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for (name, bi), ax in zip(RICH_BLOCKS, axes.ravel()):
        P = pca2(feats[:, bi*BLOCK:(bi+1)*BLOCK])
        scatter(ax, P, f"{name} aggregation")
    handles = [Line2D([], [], marker="o", ls="", color=col, label=c.replace("TCGA-", ""))
               for c, col in CANCER_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Aggregation method changes the embedding structure (PCA, by cancer type)",
                 fontsize=13, fontweight="bold")
    save(fig, os.path.join(PCA_DIR, "pca_methods_grid.png"))


# ============================================================ Part 2
def _sample_type_from_barcode(fn):
    """TCGA barcode sample-type code: 01-09 = tumor, 10-19 = normal."""
    parts = str(fn).split("-")
    if len(parts) >= 4 and parts[3][:2].isdigit():
        return "normal" if int(parts[3][:2]) >= 10 else "tumor"
    return "tumor"


def part2_chol_sweep():
    print("Part 2: CHOL-59 full method sweep by tumor/normal ...")
    files = sorted(glob.glob(os.path.join(CHOL_DIR, "*.csv")))
    methods = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f, index_col=0)
        types = np.array([_sample_type_from_barcode(idx) for idx in df.index])
        X = df.values.astype(np.float32)
        if X.shape[0] < 5 or X.shape[1] < 3:
            continue
        P = pca2(X)
        methods.append((name, P, types))
        fig, ax = plt.subplots(figsize=(4.6, 4))
        for lab, col in [("tumor", "#0d9488"), ("normal", "#c2410c")]:
            m = types == lab
            if m.any():
                ax.scatter(P[m, 0], P[m, 1], s=18, c=col, alpha=0.7,
                           edgecolor="white", linewidth=0.3, label=lab)
        ax.set_title(f"{name} (CHOL, 59 slides)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8)
        save(fig, os.path.join(CHOL_OUT, f"pca_{name}.png"))
    print(f"  {len(methods)} method scatters written")

    # one big grid of a curated subset for the page
    subset = ["mean", "median", "std", "var", "geometric_mean", "harmonic_mean",
              "trimmed_mean", "winsorized_mean", "iqr", "mad", "skewness", "kurtosis",
              "entropy", "max", "min", "percentile_50", "top10_mean", "mega_full"]
    chosen = [m for m in methods if m[0] in subset]
    chosen.sort(key=lambda m: subset.index(m[0]) if m[0] in subset else 99)
    ncol = 6; nrow = int(np.ceil(len(chosen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2.5, nrow*2.4))
    for ax in axes.ravel():
        ax.axis("off")
    for (name, P, types), ax in zip(chosen, axes.ravel()):
        ax.axis("on")
        for lab, col in [("tumor", "#0d9488"), ("normal", "#c2410c")]:
            m = types == lab
            if m.any():
                ax.scatter(P[m, 0], P[m, 1], s=8, c=col, alpha=0.7, edgecolor="none")
        ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    handles = [Line2D([], [], marker="o", ls="", color="#0d9488", label="tumor"),
               Line2D([], [], marker="o", ls="", color="#c2410c", label="normal")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle("Comparing aggregation methods on CHOL (PCA, tumor vs normal)",
                 fontsize=13, fontweight="bold")
    save(fig, os.path.join(PCA_DIR, "pca_chol_methods_grid.png"))


# ============================================================ Part 3
def part3_clustering():
    print("Part 3: clustering + FDR figures ...")
    # silhouette vs k
    clust = json.load(open(CLUST_JSON))
    methods_to_show = [m for m in ["mean", "median", "std", "mega_full", "entropy"] if m in clust]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for m in methods_to_show:
        sil = clust[m].get("silhouette_scores", {})
        ks = sorted(int(k) for k in sil)
        ax.plot(ks, [sil[str(k)] for k in ks], "-o", ms=4, lw=1.6, label=m)
    ax.set_title("Silhouette score vs number of clusters k\n(peak = natural #clusters)")
    ax.set_xlabel("k (number of clusters)"); ax.set_ylabel("silhouette score")
    ax.legend(fontsize=8)
    save(fig, os.path.join(MODELS_DIR, "clustering_silhouette.png"))

    # cluster scatter on CHOL mean embedding
    df = pd.read_csv(os.path.join(CHOL_DIR, "mean.csv"), index_col=0)
    X = StandardScaler().fit_transform(df.values)
    P = PCA(n_components=2, random_state=0).fit_transform(X)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(P)
    sil = silhouette_score(P, km.labels_)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for lab, col in [(0, "#0d9488"), (1, "#f59e0b")]:
        m = km.labels_ == lab
        ax.scatter(P[m, 0], P[m, 1], s=24, c=col, alpha=0.75,
                   edgecolor="white", linewidth=0.3, label=f"cluster {lab+1}")
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               s=180, marker="X", c="#16241d", edgecolor="white", linewidth=1, zorder=5)
    ax.set_title(f"K-means (k=2) on CHOL mean embeddings\nsilhouette = {sil:.2f}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8)
    save(fig, os.path.join(MODELS_DIR, "clustering_scatter.png"))

    # FDR summary: clusters track tissue type (chi-square) but not survival (log-rank)
    stats = pd.read_csv(STATS_CSV)
    km = stats[stats["clustering_method"] == "kmeans"]
    summary = []
    for tt, label in [("chi_square", "Tissue type\n(chi-square)"),
                      ("log_rank", "Survival\n(log-rank)")]:
        sub = km[km["test_type"] == tt]
        n_total = len(sub)
        n_sig = int(sub["significant_fdr"].sum())
        summary.append((label, n_sig, n_total))
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    labels = [s[0] for s in summary]
    sig = [s[1] for s in summary]
    tot = [s[2] for s in summary]
    x = np.arange(len(labels))
    ax.bar(x, tot, color="#e3ebe7", label="methods tested")
    ax.bar(x, sig, color="#0d9488", label="significant after FDR")
    for i, (s, t) in enumerate(zip(sig, tot)):
        ax.text(i, t + 0.5, f"{s}/{t}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("number of aggregation methods")
    ax.set_title("After FDR correction (Benjamini–Hochberg)\nclusters track tissue type, not survival")
    ax.legend(fontsize=8, loc="upper center")
    ax.set_ylim(0, max(tot) + 10)
    save(fig, os.path.join(MODELS_DIR, "fdr_significance.png"))


def main():
    part1_bycancer()
    part2_chol_sweep()
    part3_clustering()
    print("\nAll PCA / clustering figures done.")


if __name__ == "__main__":
    main()
