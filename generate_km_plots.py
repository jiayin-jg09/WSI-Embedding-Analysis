#!/usr/bin/env python3
"""Generate Kaplan-Meier plots for cancers with real WSI survival signal.

Reads ./results_v2/survival_loo/risk_scores.csv, median-splits the held-out
risk scores per (cancer, model), and plots two KM curves (high vs low risk)
with at-risk counts and log-rank p-value. Output: ./figures/km_<cancer>.png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


# (cancer, model, label) — the 4 cancers with the biggest WSI gain over
# clinical baseline, paired with their best-performing WSI model.
PLOTS = [
    ("TCGA-ACC",  "Cox_WSI_plus_Clin", "TCGA-ACC · Cox WSI+Clin · C=0.84"),
    ("TCGA-LIHC", "Cox_WSI_plus_Clin", "TCGA-LIHC · Cox WSI+Clin · C=0.69"),
    ("TCGA-CESC", "CoxnetLasso",       "TCGA-CESC · CoxnetLasso · C=0.70"),
    ("TCGA-STAD", "CoxnetLasso",       "TCGA-STAD · CoxnetLasso · C=0.60"),
]

# Pine theme colors so plots match the website palette
BG = "#16201b"
PANEL_EDGE = "#1f2c25"
TEXT = "#d6e4dc"
MUTED = "#8aa39a"
HIGH = "#f8a07b"   # warn / high-risk
LOW = "#5eead4"    # accent / low-risk


def km_plot(ax, df_sub: pd.DataFrame, title: str) -> None:
    # Collapse to one row per (participant_id, time, event, risk) since the
    # pipeline broadcasts risk per slide — we want one observation per patient.
    df = (df_sub.dropna(subset=["risk_score"])
                .groupby("participant_id", as_index=False)
                .agg(risk_score=("risk_score", "mean"),
                     time=("time", "first"),
                     event=("event", "first")))

    median = df["risk_score"].median()
    high = df["risk_score"] > median
    low = ~high

    p_value = np.nan
    try:
        lr = logrank_test(
            df.loc[high, "time"], df.loc[low, "time"],
            event_observed_A=df.loc[high, "event"],
            event_observed_B=df.loc[low, "event"],
        )
        p_value = lr.p_value
    except Exception:
        pass

    def _plot(mask, color, label):
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=label)
        kmf.plot_survival_function(
            ax=ax, ci_show=True, color=color, linewidth=2.0,
        )

    _plot(high, HIGH, f"High risk (n={int(high.sum())})")
    _plot(low,  LOW,  f"Low risk (n={int(low.sum())})")

    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=11, pad=10)
    ax.set_xlabel("Days", color=MUTED, fontsize=9)
    ax.set_ylabel("Survival probability", color=MUTED, fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
    ax.grid(True, alpha=0.15, linestyle="--", color=MUTED)

    p_str = f"log-rank p = {p_value:.1e}" if not np.isnan(p_value) else "log-rank p = n/a"
    ax.text(0.97, 0.05, p_str, transform=ax.transAxes,
            ha="right", va="bottom", color=TEXT, fontsize=9,
            bbox=dict(facecolor=BG, edgecolor=PANEL_EDGE, boxstyle="round,pad=0.3"))
    leg = ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor(PANEL_EDGE)
    for txt in leg.get_texts():
        txt.set_color(TEXT)


def main() -> None:
    rs_path = Path("./results_v2/survival_loo/risk_scores.csv")
    if not rs_path.exists():
        raise SystemExit(f"Missing {rs_path}")
    rs = pd.read_csv(rs_path)

    os.makedirs("figures", exist_ok=True)

    # Combined 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=BG)
    for (cancer, model, title), ax in zip(PLOTS, axes.flat):
        sub = rs[(rs["cancer_type"] == cancer) & (rs["model"] == model)]
        if sub.empty:
            ax.text(0.5, 0.5, f"No data: {cancer} / {model}",
                    transform=ax.transAxes, ha="center", color=MUTED)
            continue
        km_plot(ax, sub, title)
    fig.suptitle("Held-out risk score stratification · pooled LOPO",
                 color=TEXT, fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_combined = Path("figures/km_combined.png")
    fig.savefig(out_combined, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_combined}")

    # Individual plots (in case we want to embed one at a time)
    for cancer, model, title in PLOTS:
        sub = rs[(rs["cancer_type"] == cancer) & (rs["model"] == model)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=BG)
        km_plot(ax, sub, title)
        fig.tight_layout()
        out = Path(f"figures/km_{cancer.lower().replace('-', '_')}.png")
        fig.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
