"""
uni2_dimension_enrichment.py
============================
Internal/optional tool: annotate each UNI2 embedding dimension by gene-set enrichment.

For every dimension we take the genes it correlates with most strongly WITHIN
cancer type (from uni2_mean_expression.parquet, residualized mode), map ENSEMBL ->
HUGO, and test those top genes for over-representation in each gene set
(hypergeometric + Benjamini-Hochberg). LOCAL OUTPUTS ONLY (not shown on the site).

Prereq: a local data/genesets.gmt (a gene-set collection in GMT format; gitignored).
Run: python uni2_dimension_enrichment.py
Output: results_v2/dim_omics/dimension_labels.csv (gitignored)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPR = os.path.join("results_v2", "dim_omics", "uni2_mean_expression.parquet")
MAP = os.path.join("..", "TCGA multiomics", "TCGA_MODULES_FULL",
                   "ENSEMBL_HUGO_MAPPING_V2.parquet")
GMT = os.path.join("data", "genesets.gmt")
OUT = os.path.join("results_v2", "dim_omics", "dimension_labels.csv")
TOPK = 150          # top correlated genes per dimension
Q_SIG = 0.05


def load_gmt(path):
    sets = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        genes = {g.strip().upper() for g in parts[2:] if g.strip()}
        if genes:
            sets[parts[0]] = genes
    return sets


def bh(p):
    p = np.asarray(p, float); n = p.size
    o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def main():
    if not os.path.exists(GMT):
        print(f"missing {GMT} — fetch the gene-set gmt first; skipping"); return
    hall = load_gmt(GMT)
    print(f"{len(hall)} gene sets", flush=True)

    # ENSEMBL(expr_...) -> HUGO
    m = pd.read_parquet(MAP)
    m = m[m["query_success"]] if "query_success" in m.columns else m
    feat2hugo = dict(zip(m["original_column"], m["hugo_symbol"].astype(str).str.upper()))

    df = pd.read_parquet(EXPR, columns=["dim", "feature", "rho", "mode"])
    df = df[df["mode"] == "residualized"].copy()
    df["hugo"] = df["feature"].map(feat2hugo)
    df = df.dropna(subset=["hugo"])
    df = df[df["hugo"].str.match(r"^[A-Z0-9-]+$")]            # drop odd symbols
    universe = set(df["hugo"].unique())
    M = len(universe)
    print(f"gene universe (mapped, residualized): {M}", flush=True)

    # pre-intersect each gene set with the universe
    hall_u = {k: (v & universe) for k, v in hall.items()}
    hall_u = {k: v for k, v in hall_u.items() if len(v) >= 5}

    df["abs"] = df["rho"].abs()
    rows = []
    for dim, g in df.groupby("dim"):
        top = set(g.nlargest(TOPK, "abs")["hugo"])
        N = len(top)
        names, ps, ks = [], [], []
        for setname, sg in hall_u.items():
            k = len(top & sg)
            if k == 0:
                continue
            p = hypergeom.sf(k - 1, M, len(sg), N)
            names.append(setname); ps.append(p); ks.append(k)
        if not names:
            rows.append({"dim": dim, "top_pathway": "none", "q_value": 1.0, "overlap": 0})
            continue
        q = bh(ps)
        j = int(np.argmin(q))
        rows.append({"dim": dim, "top_pathway": names[j] if q[j] < Q_SIG else "none",
                     "q_value": round(float(q[j]), 5), "overlap": ks[j]})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    n_lab = int((out["top_pathway"] != "none").sum())
    print(f"wrote {OUT}: {n_lab}/{len(out)} dimensions labeled (q<{Q_SIG})", flush=True)

    # figure: how many dimensions map to each program (top 15)
    vc = out[out["top_pathway"] != "none"]["top_pathway"].value_counts().head(15)[::-1]
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(vc)), vc.values, color="#0d9488")
    ax.set_yticks(range(len(vc))); ax.set_yticklabels(vc.index, fontsize=8)
    ax.set_xlabel("number of UNI2 dimensions whose top program is this gene set")
    ax.set_title("Biological vocabulary of the UNI2 embedding\n"
                 "(dimensions labeled by within-cancer gene enrichment)", fontweight="bold")
    fig.tight_layout()
    figout = os.path.join("results_v2", "dim_omics", "uni2_dimension_pathways.png")
    fig.savefig(figout, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {figout}", flush=True)
    print("\nTop programs:")
    print(out[out["top_pathway"] != "none"]["top_pathway"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
