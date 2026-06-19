"""
uni2_vs_ctranspath.py
=====================
Do two different pathology foundation models encode the SAME biology?

UNI2 (this project) and CTransPath (the mentor's dim_omic_correlations/) were both
correlated, dimension-by-dimension, against the same TCGA molecular features. Their
embedding dimensions are not comparable one-to-one (UNI2 has 1,536, CTransPath 768),
so we compare at the FEATURE level: for each molecular feature, how strongly does the
*best* dimension of each model encode it (max |rho| over dims)? If the two models'
per-feature "encodability" agrees, they capture the same underlying biology.

Comparison is pooled-vs-pooled (the CTransPath files are pooled only). Output:
  results_v2/dim_omics/compare_<omic>.csv  — per-feature best |rho| for both models
  results_v2/dim_omics/compare_summary.csv — per-omic Spearman agreement + overlap
  figures/uni2_vs_ctranspath.png           — scatter grid, one panel per omic

Run: python uni2_vs_ctranspath.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

UNI2_DIR = os.path.join("results_v2", "dim_omics")
CTP_DIR = os.path.join("..", "TCGA multiomics", "dim_omic_correlations")
RPPA_MAP = os.path.join("..", "TCGA multiomics", "TCGA_MODULES_FULL", "ARCHIVE",
                        "RPPA_numeric_column_mapping.csv")
OMICS = ["immune", "immune_signatures", "rppa", "mutations", "cnv", "expression"]
TOPK = 50   # feature-overlap among each model's top-K most-encodable features

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 9, "axes.grid": True, "grid.color": "#e3ebe7"})


def norm_key(feature, omic, rppa_map):
    """Normalize a feature name to a model-agnostic join key."""
    if omic == "mutations":
        return feature.replace("mut_", "").replace("bin_", "")
    if omic == "rppa":
        if feature.startswith("protein."):            # CTransPath raw column
            return rppa_map.get(feature, feature)
        return feature                                # UNI2 already mapped (rppa_NAME)
    return feature                                    # immune/sig/cnv/expr already match


def best_per_feature(path, omic, rppa_map, pooled_only):
    df = pd.read_parquet(path, columns=["feature", "rho", "mode"]
                         if pooled_only else ["feature", "rho"])
    if pooled_only and "mode" in df.columns:
        df = df[df["mode"] == "pooled"]
    df["key"] = [norm_key(f, omic, rppa_map) for f in df["feature"]]
    return df.groupby("key")["rho"].apply(lambda s: s.abs().max())


def biology_figure(rppa_map):
    """Bar chart: how strongly UNI2 encodes each omic, pooled vs within-cancer.

    Per omic, we take each feature's best |rho| over dims, then the MEDIAN across
    features (a typical-feature encodability, not the single best). Shows the
    cancer-confound gap and that microenvironment/expression >> mutations."""
    rows = []
    for omic in OMICS:
        up = os.path.join(UNI2_DIR, f"uni2_mean_{omic}.parquet")
        if not os.path.exists(up):
            continue
        df = pd.read_parquet(up, columns=["feature", "rho", "mode"])
        rec = {"omic": omic}
        for mode in ("pooled", "residualized"):
            d = df[df["mode"] == mode]
            best = d.groupby("feature")["rho"].apply(lambda s: s.abs().max())
            rec[mode] = float(best.median())
        rows.append(rec)
    s = pd.DataFrame(rows).sort_values("residualized", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(s))
    ax.barh(y - 0.2, s["pooled"], height=0.4, color="#94a3b8",
            label="pooled (cancer-type confounded)")
    ax.barh(y + 0.2, s["residualized"], height=0.4, color="#0d9488",
            label="within-cancer (residualized)")
    ax.set_yticks(y); ax.set_yticklabels(s["omic"])
    ax.set_xlabel("typical feature encodability  (median best |rho| over dims)")
    ax.set_title("What biology do UNI2 histology embeddings encode?", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = os.path.join("figures", "uni2_biology_encoded.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main():
    rppa_map = {}
    if os.path.exists(RPPA_MAP):
        m = pd.read_csv(RPPA_MAP)
        rppa_map = {r["rppa_column"]: "rppa_" + str(r["protein_name"]).lstrip("X")
                    for _, r in m.iterrows()}

    summ, panels = [], []
    for omic in OMICS:
        up = os.path.join(UNI2_DIR, f"uni2_mean_{omic}.parquet")
        cp = os.path.join(CTP_DIR, f"ctranspath_mean_{omic}.parquet")
        if not (os.path.exists(up) and os.path.exists(cp)):
            print(f"skip {omic}: missing file"); continue
        u = best_per_feature(up, omic, rppa_map, pooled_only=True).rename("uni2")
        c = best_per_feature(cp, omic, rppa_map, pooled_only=False).rename("ctranspath")
        j = pd.concat([u, c], axis=1).dropna()
        if len(j) < 10:
            print(f"skip {omic}: only {len(j)} shared features"); continue
        rho, _ = spearmanr(j["uni2"], j["ctranspath"])
        top_u = set(j["uni2"].sort_values(ascending=False).head(TOPK).index)
        top_c = set(j["ctranspath"].sort_values(ascending=False).head(TOPK).index)
        overlap = len(top_u & top_c)
        j.to_csv(os.path.join(UNI2_DIR, f"compare_{omic}.csv"))
        summ.append({"omic": omic, "n_features": len(j),
                     "spearman_agreement": round(float(rho), 3),
                     f"top{TOPK}_overlap": overlap})
        panels.append((omic, j, rho))
        print(f"{omic:18s} {len(j):6d} shared features | Spearman agreement = {rho:.3f} "
              f"| top-{TOPK} overlap {overlap}/{TOPK}", flush=True)

    if summ:
        pd.DataFrame(summ).to_csv(os.path.join(UNI2_DIR, "compare_summary.csv"), index=False)

    # scatter grid
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.3))
    for ax in np.ravel(axes):
        ax.axis("off")
    for (omic, j, rho), ax in zip(panels, np.ravel(axes)):
        ax.axis("on")
        ax.scatter(j["ctranspath"], j["uni2"], s=6, c="#0d9488", alpha=0.35,
                   edgecolor="none")
        lim = max(j["uni2"].max(), j["ctranspath"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="#c2410c", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(f"{omic}\nagreement ρ={rho:.2f}", fontsize=10)
        ax.set_xlabel("CTransPath best |rho|"); ax.set_ylabel("UNI2 best |rho|")
    fig.suptitle("Do UNI2 and CTransPath encode the same biology?\n"
                 "per-feature best |correlation| across embedding dimensions (pooled)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join("figures", "uni2_vs_ctranspath.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}", flush=True)
    if summ:
        print(pd.DataFrame(summ).to_string(index=False))

    biology_figure(rppa_map)


if __name__ == "__main__":
    main()
