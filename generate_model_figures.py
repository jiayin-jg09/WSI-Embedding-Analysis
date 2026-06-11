"""
generate_model_figures.py
==========================
Illustrative example plots for the website's "Models explained" pages.

Each plot is a small, clearly-labeled schematic built from tiny synthetic data
(no real cohort data, no H5 files) so it is thermally trivial and reproducible.
Outputs land in figures/models/*.png on a white background so they read on both
the light and dark site themes.

Run:  python generate_model_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.datasets import make_classification, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

OUT = os.path.join("figures", "models")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.RandomState(42)

# --- shared style --------------------------------------------------------
ACCENT = "#0d9488"
ACCENT2 = "#5eead4"
WARN = "#c2410c"
MUTED = "#5a6f66"
GRID = "#d6e1db"
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#b9c8c1",
    "axes.labelcolor": "#16241d",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "text.color": "#16241d",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# shared 2-class 2-D set for decision-boundary plots
Xc, yc = make_classification(n_samples=160, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1,
                             class_sep=1.1, random_state=7)


def boundary(clf, title, name, proba=True):
    clf.fit(Xc, yc)
    x_min, x_max = Xc[:, 0].min() - 0.8, Xc[:, 0].max() + 0.8
    y_min, y_max = Xc[:, 1].min() - 0.8, Xc[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if proba and hasattr(clf, "predict_proba"):
        Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    elif hasattr(clf, "decision_function"):
        Z = clf.decision_function(grid).reshape(xx.shape)
    else:
        Z = clf.predict(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5, 3.7))
    ax.contourf(xx, yy, Z, levels=20, cmap="BrBG", alpha=0.55)
    ax.contour(xx, yy, Z, levels=[0.5] if (proba and hasattr(clf, "predict_proba"))
               else [0], colors="#16241d", linewidths=1.4)
    ax.scatter(Xc[yc == 0, 0], Xc[yc == 0, 1], s=22, c="#b45309",
               edgecolor="white", linewidth=0.4, label="class 0")
    ax.scatter(Xc[yc == 1, 0], Xc[yc == 1, 1], s=22, c=ACCENT,
               edgecolor="white", linewidth=0.4, label="class 1")
    ax.set_title(title)
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    save(fig, name)


# =========================================================================
# CLASSIFICATION
# =========================================================================
def fig_knn():
    boundary(KNeighborsClassifier(n_neighbors=5),
             "k-Nearest Neighbours (k = 5)\nlabel = majority vote of nearest points",
             "knn.png")


def fig_logreg():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    # sigmoid
    z = np.linspace(-6, 6, 200)
    axes[0].plot(z, 1 / (1 + np.exp(-z)), color=ACCENT, lw=2.2)
    axes[0].axhline(0.5, color=MUTED, ls=":", lw=1)
    axes[0].axvline(0, color=MUTED, ls=":", lw=1)
    axes[0].set_title("The logistic (sigmoid) link")
    axes[0].set_xlabel("weighted sum of features"); axes[0].set_ylabel("P(class 1)")
    # boundary inset
    clf = LogisticRegression().fit(Xc, yc)
    xx, yy = np.meshgrid(np.linspace(Xc[:, 0].min()-.8, Xc[:, 0].max()+.8, 200),
                         np.linspace(Xc[:, 1].min()-.8, Xc[:, 1].max()+.8, 200))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    axes[1].contourf(xx, yy, Z, levels=20, cmap="BrBG", alpha=0.55)
    axes[1].contour(xx, yy, Z, levels=[0.5], colors="#16241d", linewidths=1.4)
    axes[1].scatter(Xc[:, 0], Xc[:, 1], s=16, c=yc, cmap="BrBG_r",
                    edgecolor="white", linewidth=0.3)
    axes[1].set_title("→ a linear decision boundary")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].grid(False)
    save(fig, "logreg.png")


def fig_regularization():
    # coefficient shrinkage / sparsity for L1 vs L2 as penalty strength grows
    rng = np.random.RandomState(3)
    p = 8
    Cs = np.logspace(2, -1.5, 30)  # decreasing C = more regularization
    Xr, yr = make_classification(n_samples=200, n_features=p, n_informative=3,
                                 n_redundant=0, random_state=3)
    l1, l2 = [], []
    for C in Cs:
        l1.append(LogisticRegression(penalty="elasticnet", solver="saga",
                                     l1_ratio=1.0, C=C, max_iter=8000)
                  .fit(Xr, yr).coef_.ravel())
        l2.append(LogisticRegression(penalty="elasticnet", solver="saga",
                                     l1_ratio=0.0, C=C, max_iter=8000)
                  .fit(Xr, yr).coef_.ravel())
    l1, l2 = np.array(l1), np.array(l2)
    strength = np.log10(1 / Cs)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    for j in range(p):
        axes[0].plot(strength, l1[:, j], lw=1.4)
        axes[1].plot(strength, l2[:, j], lw=1.4)
    axes[0].set_title("L1 (Lasso): coefficients hit exactly 0")
    axes[1].set_title("L2 (Ridge): coefficients shrink smoothly")
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("more regularization →")
    axes[0].set_ylabel("coefficient value")
    save(fig, "regularization_l1_l2.png")


def fig_linearsvc():
    clf = LinearSVC(C=1.0, max_iter=5000).fit(Xc, yc)
    w = clf.coef_[0]; b = clf.intercept_[0]
    xs = np.linspace(Xc[:, 0].min()-.8, Xc[:, 0].max()+.8, 100)
    ys = -(w[0] * xs + b) / w[1]
    margin = 1 / np.linalg.norm(w)
    off = margin * np.sqrt(1 + (w[0]/w[1])**2)
    fig, ax = plt.subplots(figsize=(5, 3.7))
    ax.plot(xs, ys, color="#16241d", lw=1.6, label="decision boundary")
    ax.plot(xs, ys + off, color=MUTED, ls="--", lw=1)
    ax.plot(xs, ys - off, color=MUTED, ls="--", lw=1, label="margins")
    ax.scatter(Xc[yc == 0, 0], Xc[yc == 0, 1], s=22, c="#b45309",
               edgecolor="white", linewidth=0.4)
    ax.scatter(Xc[yc == 1, 0], Xc[yc == 1, 1], s=22, c=ACCENT,
               edgecolor="white", linewidth=0.4)
    ax.set_ylim(Xc[:, 1].min()-.8, Xc[:, 1].max()+.8)
    ax.set_title("Linear SVM\nwidest margin between the classes")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "linearsvc.png")


def fig_random_forest():
    boundary(RandomForestClassifier(n_estimators=200, max_depth=None, random_state=1),
             "Random Forest\nmany bootstrapped trees, averaged", "random_forest.png")


def fig_extratrees():
    boundary(ExtraTreesClassifier(n_estimators=200, random_state=1),
             "Extremely Randomized Trees\nrandom split thresholds → smoother, faster",
             "extratrees.png")


def fig_gradient_boosting():
    boundary(GradientBoostingClassifier(n_estimators=120, max_depth=2,
                                        learning_rate=0.2, random_state=1),
             "Gradient Boosting\ntrees added one-by-one to fix residual errors",
             "gradient_boosting.png")


def fig_lda():
    boundary(LinearDiscriminantAnalysis(),
             "Linear Discriminant Analysis\none Gaussian per class, shared shape",
             "lda.png", proba=True)


def fig_gaussiannb():
    boundary(GaussianNB(),
             "Gaussian Naive Bayes\nper-feature Gaussians, assumed independent",
             "gaussiannb.png", proba=True)


def fig_roc_auc():
    Xr, yr = make_classification(n_samples=300, n_features=6, n_informative=3,
                                 random_state=5)
    scores = LogisticRegression(max_iter=1000).fit(Xr, yr).predict_proba(Xr)[:, 1]
    fpr, tpr, _ = roc_curve(yr, scores)
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot(fpr, tpr, color=ACCENT, lw=2.4, label=f"model (AUC = {a:.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="chance (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
    ax.set_title("ROC curve & AUC")
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "roc_auc.png")


def fig_pca():
    X3, y3 = make_blobs(n_samples=240, centers=3, n_features=6,
                        cluster_std=2.4, random_state=2)
    Z = PCA(n_components=2).fit(X3)
    P = Z.transform(X3)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    axes[0].scatter(P[:, 0], P[:, 1], s=16, c=y3, cmap="viridis",
                    edgecolor="white", linewidth=0.3)
    axes[0].set_title("Data projected onto top 2 PCs")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].grid(False)
    pcs = PCA().fit(make_blobs(n_samples=240, centers=3, n_features=20,
                               cluster_std=2.4, random_state=2)[0])
    cum = np.cumsum(pcs.explained_variance_ratio_)
    axes[1].plot(range(1, len(cum)+1), cum, "-o", color=ACCENT, ms=3, lw=1.6)
    axes[1].axhline(0.95, color=WARN, ls="--", lw=1, label="95% variance")
    axes[1].set_title("Variance retained vs #components")
    axes[1].set_xlabel("number of components"); axes[1].set_ylabel("cumulative variance")
    axes[1].legend(fontsize=8)
    save(fig, "pca.png")


# =========================================================================
# SURVIVAL — concepts
# =========================================================================
def fig_censoring():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    rng = np.random.RandomState(1)
    n = 8
    times = rng.uniform(2, 10, n)
    event = rng.rand(n) > 0.45
    for i in range(n):
        ax.plot([0, times[i]], [i, i], color=MUTED, lw=2, solid_capstyle="round")
        if event[i]:
            ax.plot(times[i], i, "X", color=WARN, ms=11, label="death (event)" if i == 0 else "")
        else:
            ax.plot(times[i], i, "o", color=ACCENT, ms=9, mfc="white", mew=2,
                    label="censored (lost / still alive)" if not event[:i].all() and i == 1 else "")
    handles = [Line2D([], [], marker="X", color="w", mfc=WARN, ms=11, label="death (event)"),
               Line2D([], [], marker="o", color="w", mfc="white", mec=ACCENT, mew=2, ms=9,
                      label="censored")]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_title("Censoring: not every patient is observed to the event")
    ax.set_xlabel("time since diagnosis"); ax.set_yticks([])
    ax.set_ylim(-1, n); ax.grid(axis="y", visible=False)
    save(fig, "censoring.png")


def km_curve(ax, t0, lam, color, label, n=120, seed=0):
    rng = np.random.RandomState(seed)
    t = rng.exponential(1/lam, n)
    cens = rng.uniform(0, t0, n)
    obs = np.minimum(t, cens)
    ev = (t <= cens)
    order = np.argsort(obs)
    obs, ev = obs[order], ev[order]
    times, surv = [0], [1.0]
    s = 1.0; at_risk = n
    for i in range(n):
        if ev[i]:
            s *= (1 - 1/at_risk)
        times.append(obs[i]); surv.append(s)
        at_risk -= 1
    ax.step(times, surv, where="post", color=color, lw=2.2, label=label)
    return times, surv


def fig_kaplan_meier():
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    km_curve(ax, 9, 0.12, ACCENT, "low predicted risk", seed=1)
    km_curve(ax, 9, 0.34, WARN, "high predicted risk", seed=2)
    ax.set_ylim(0, 1.02)
    ax.set_title("Kaplan–Meier: survival of risk groups\n(log-rank tests the gap)")
    ax.set_xlabel("time"); ax.set_ylabel("survival probability")
    ax.legend(fontsize=8)
    save(fig, "kaplan_meier.png")


def fig_c_index():
    rng = np.random.RandomState(4)
    n = 60
    risk = rng.rand(n)
    time = 10 - 7 * risk + rng.normal(0, 1.1, n)  # higher risk -> shorter time
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.scatter(risk, time, s=22, c=ACCENT, edgecolor="white", linewidth=0.4)
    # highlight a concordant pair
    i, j = 5, 40
    ax.plot([risk[i], risk[j]], [time[i], time[j]], color=WARN, lw=1.6, zorder=1)
    ax.scatter([risk[i], risk[j]], [time[i], time[j]], s=60, facecolors="none",
               edgecolors=WARN, linewidths=1.8, zorder=3)
    ax.set_title("Concordance (C-index)\nhigher predicted risk → shorter survival?")
    ax.set_xlabel("predicted risk score"); ax.set_ylabel("observed survival time")
    ax.text(0.02, 0.04, "C-index = fraction of patient pairs\nranked in the correct order",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    save(fig, "c_index.png")


def fig_hazard_ratio():
    # real per-SD HRs from the project (best WSI model per cancer)
    labels = ["ACC", "CESC", "LIHC", "COAD", "STAD", "READ"]
    hr = [6.88, 2.13, 1.95, 1.42, 1.40, 0.88]
    lo = [4.5, 1.5, 1.6, 1.05, 1.08, 0.62]
    hi = [10.5, 3.0, 2.4, 1.9, 1.8, 1.25]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for yi, h, l, hgh in zip(y, hr, lo, hi):
        c = ACCENT if l > 1 else WARN
        ax.plot([l, hgh], [yi, yi], color=c, lw=2)
        ax.plot(h, yi, "o", color=c, ms=7)
    ax.axvline(1.0, color=MUTED, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xticks([0.7, 1, 2, 4, 8]); ax.set_xticklabels(["0.7", "1", "2", "4", "8"])
    ax.set_title("Hazard ratio per 1 SD of risk\n(>1 = higher risk → worse survival)")
    ax.set_xlabel("hazard ratio (log scale)")
    ax.grid(axis="y", visible=False)
    save(fig, "hazard_ratio.png")


# =========================================================================
# SURVIVAL — models
# =========================================================================
def fig_cox_ph():
    t = np.linspace(0, 10, 200)
    S0 = np.exp(-0.12 * t)            # baseline survival
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for hr, c, lab in [(0.5, ACCENT, "HR = 0.5 (protective)"),
                       (1.0, MUTED, "baseline"),
                       (2.0, WARN, "HR = 2 (higher risk)")]:
        ax.plot(t, S0 ** hr, color=c, lw=2.2, label=lab)
    ax.set_ylim(0, 1.02)
    ax.set_title("Cox proportional hazards\ncovariates scale one baseline curve")
    ax.set_xlabel("time"); ax.set_ylabel("survival probability")
    ax.legend(fontsize=8)
    save(fig, "cox_ph.png")


def _coef_paths(l1_ratio, seed=0):
    rng = np.random.RandomState(seed)
    p = 10
    alphas = np.logspace(-2.3, 0.2, 40)
    true = np.zeros(p); true[[1, 4, 7]] = [1.6, -1.2, 0.9]
    paths = []
    for a in alphas:
        # soft-threshold style shrinkage toward zero (illustrative, not a real fit)
        shrink = a * (l1_ratio * 1.0)
        ridge = 1 / (1 + a * (1 - l1_ratio) * 6)
        coef = np.sign(true) * np.maximum(np.abs(true) - shrink, 0) * ridge
        # small noise coefs that lasso kills fast
        noise = rng.normal(0, 0.25, p)
        noise[[1, 4, 7]] = 0
        coef = coef + np.sign(noise) * np.maximum(np.abs(noise) - shrink*1.6, 0) * ridge
        paths.append(coef)
    return alphas, np.array(paths)


def fig_coxnet_lasso():
    alphas, paths = _coef_paths(l1_ratio=1.0, seed=1)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for j in range(paths.shape[1]):
        ax.plot(np.log10(alphas), paths[:, j], lw=1.5)
    ax.axhline(0, color=MUTED, ls=":", lw=0.9)
    ax.set_title("Coxnet — Lasso (L1)\nmost coefficients driven to exactly 0")
    ax.set_xlabel("log10(penalty α)  →  stronger"); ax.set_ylabel("coefficient")
    save(fig, "coxnet_lasso.png")


def fig_coxnet_elasticnet():
    alphas, paths = _coef_paths(l1_ratio=0.5, seed=1)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for j in range(paths.shape[1]):
        ax.plot(np.log10(alphas), paths[:, j], lw=1.5)
    ax.axhline(0, color=MUTED, ls=":", lw=0.9)
    ax.set_title("Coxnet — ElasticNet (L1+L2)\nshrinks together, keeps correlated groups")
    ax.set_xlabel("log10(penalty α)  →  stronger"); ax.set_ylabel("coefficient")
    save(fig, "coxnet_elasticnet.png")


def fig_rsf():
    t = np.linspace(0, 10, 120)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    rng = np.random.RandomState(6)
    curves = []
    for k in range(12):
        lam = 0.10 + 0.06 * rng.rand()
        steps = np.exp(-lam * t) + rng.normal(0, 0.015, t.size)
        steps = np.clip(np.minimum.accumulate(steps), 0, 1)
        curves.append(steps)
        ax.step(t, steps, where="post", color=ACCENT2, lw=0.9, alpha=0.6)
    ax.step(t, np.mean(curves, axis=0), where="post", color=ACCENT, lw=2.6,
            label="forest average")
    ax.set_ylim(0, 1.02)
    ax.set_title("Random Survival Forest\nmany tree survival curves, averaged")
    ax.set_xlabel("time"); ax.set_ylabel("survival probability")
    ax.legend(fontsize=8)
    save(fig, "rsf.png")


def fig_gbsurv():
    stages = np.arange(1, 51)
    cindex = 0.5 + 0.34 * (1 - np.exp(-stages / 12)) - 0.0006 * stages
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(stages, cindex, color=ACCENT, lw=2.4)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_ylim(0.48, 0.9)
    ax.set_title("Gradient-Boosted Survival\neach added tree corrects the last")
    ax.set_xlabel("number of boosting stages"); ax.set_ylabel("training C-index")
    save(fig, "gbsurv.png")


# =========================================================================
# AGGREGATION
# =========================================================================
def fig_agg_summary_stats():
    rng = np.random.RandomState(8)
    vals = np.concatenate([rng.normal(-0.6, 0.5, 700), rng.normal(0.8, 0.4, 500)])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(vals, bins=40, color="#cdeee8", edgecolor="#9ed8cf")
    pcts = {"P10": 10, "P25": 25, "P75": 75, "P90": 90}
    for lab, q in pcts.items():
        x = np.percentile(vals, q)
        ax.axvline(x, color=MUTED, ls=":", lw=1)
        ax.text(x, ax.get_ylim()[1]*0.92, lab, rotation=90, fontsize=7,
                color=MUTED, ha="right", va="top")
    m = vals.mean(); s = vals.std()
    ax.axvline(m, color=ACCENT, lw=2.2, label=f"mean = {m:.2f}")
    ax.axvspan(m - s, m + s, color=ACCENT, alpha=0.10, label=f"±1 std = {s:.2f}")
    ax.set_title("Aggregating one embedding dimension across a slide's patches")
    ax.set_xlabel("value of feature f_k over all patches"); ax.set_ylabel("patch count")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "agg_summary_stats.png")


def fig_agg_rich_vector():
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.axis("off")
    stats = ["mean", "std", "P10", "P25", "P75", "P90"]
    colors = ["#0d9488", "#14b8a6", "#2dd4bf", "#5eead4", "#67e8f9", "#38bdf8"]
    ax.text(0.04, 0.9, "N patches × 1,536 dims", fontsize=10, color=MUTED)
    ax.add_patch(plt.Rectangle((0.04, 0.52), 0.16, 0.3, fc="#e6f4f1", ec="#9ed8cf"))
    ax.text(0.12, 0.67, "patch\nembeddings", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(0.295, 0.67), xytext=(0.215, 0.67),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))
    x0, w, gap = 0.30, 0.105, 0.012
    x = x0
    for s, c in zip(stats, colors):
        ax.add_patch(plt.Rectangle((x, 0.52), w, 0.3, fc=c, ec="white"))
        ax.text(x + w/2, 0.67, s, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        x += w + gap
    x_end = x - gap
    ax.text(x0, 0.45, "each block = 1,536 numbers", fontsize=8, color=MUTED)
    # bracket under the blocks
    ax.plot([x0, x_end], [0.36, 0.36], color=MUTED, lw=1)
    ax.plot([x0, x0], [0.36, 0.39], color=MUTED, lw=1)
    ax.plot([x_end, x_end], [0.36, 0.39], color=MUTED, lw=1)
    ax.annotate("", xy=((x0+x_end)/2, 0.27), xytext=((x0+x_end)/2, 0.36),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    ax.text((x0 + x_end)/2, 0.22, "concatenated → one 9,216-dim slide vector",
            ha="center", fontsize=10, color="#16241d", fontweight="bold")
    ax.set_title("The \"rich\" aggregation: 6 statistics → fixed-length slide vector",
                 fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0.15, 1)
    save(fig, "agg_rich_vector.png")


def main():
    # classification
    fig_knn(); fig_logreg(); fig_regularization(); fig_linearsvc()
    fig_random_forest(); fig_extratrees(); fig_gradient_boosting()
    fig_lda(); fig_gaussiannb(); fig_roc_auc(); fig_pca()
    # survival concepts
    fig_censoring(); fig_kaplan_meier(); fig_c_index(); fig_hazard_ratio()
    # survival models
    fig_cox_ph(); fig_coxnet_lasso(); fig_coxnet_elasticnet(); fig_rsf(); fig_gbsurv()
    # aggregation
    fig_agg_summary_stats(); fig_agg_rich_vector()
    print("\nAll model figures written to", OUT)


if __name__ == "__main__":
    main()
