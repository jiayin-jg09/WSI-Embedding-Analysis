#!/usr/bin/env python3
"""
Generate Workflow Figures for WSI Embedding Analysis Pipeline

Creates 4 publication-quality figures illustrating the pipeline:
  1. Pipeline Overview (end-to-end flow)
  2. Aggregation Methods Taxonomy (43 methods by category)
  3. Modeling Pipeline Detail (classification, survival, clustering)
  4. Results & Outputs (file structure and example heatmap)

Usage:
    python generate_workflow_figures.py [--output-dir ./figures]

Dependencies: matplotlib, numpy, seaborn (included in requirements.txt)
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as path_effects
import seaborn as sns

# =============================================================================
# Style constants (matching JiaYin project style)
# =============================================================================
COLORS = {
    'blue': '#2980b9',
    'dark_blue': '#1a5276',
    'green': '#27ae60',
    'dark_green': '#1e8449',
    'orange': '#f39c12',
    'red': '#e74c3c',
    'purple': '#8e44ad',
    'teal': '#16a085',
    'gray': '#7f8c8d',
    'light_gray': '#ecf0f1',
    'dark_gray': '#2c3e50',
    'white': '#ffffff',
    'light_blue': '#d4e6f1',
    'light_green': '#d5f5e3',
    'light_orange': '#fdebd0',
    'light_red': '#fadbd8',
    'light_purple': '#e8daef',
    'light_teal': '#d1f2eb',
}

DPI = 150
FONT_FAMILY = 'DejaVu Sans'


def draw_rounded_box(ax, x, y, width, height, text, color, text_color='white',
                     fontsize=10, fontweight='bold', alpha=0.95, text_lines=None):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor='white', linewidth=1.5, alpha=alpha,
        zorder=3
    )
    ax.add_patch(box)

    if text_lines:
        line_height = fontsize * 0.0022
        total_height = (len(text_lines) - 1) * line_height
        start_y = y + height / 2 + total_height / 2
        for i, line in enumerate(text_lines):
            fs = fontsize if i == 0 else fontsize - 1
            fw = fontweight if i == 0 else 'normal'
            ax.text(x + width / 2, start_y - i * line_height, line,
                    ha='center', va='center', fontsize=fs, fontweight=fw,
                    color=text_color, zorder=4, family=FONT_FAMILY)
    else:
        ax.text(x + width / 2, y + height / 2, text,
                ha='center', va='center', fontsize=fontsize, fontweight=fontweight,
                color=text_color, zorder=4, family=FONT_FAMILY)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color='#7f8c8d', style='->', linewidth=2):
    """Draw a curved arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle="arc3,rad=0.0",
        arrowstyle=f"{style},head_width=0.008,head_length=0.006",
        color=color, linewidth=linewidth, zorder=2
    )
    ax.add_patch(arrow)
    return arrow


# =============================================================================
# FIGURE 1: Pipeline Overview
# =============================================================================
def create_fig1_pipeline_overview(output_dir: Path):
    """End-to-end pipeline flow diagram."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.96, 'WSI Embedding Analysis Pipeline',
            ha='center', va='top', fontsize=22, fontweight='bold',
            color=COLORS['dark_gray'], family=FONT_FAMILY)
    ax.text(0.5, 0.92, 'From Whole Slide Images to Biomarker Discovery',
            ha='center', va='top', fontsize=13, color=COLORS['gray'],
            family=FONT_FAMILY)

    # === Row 1: Data Acquisition (top) ===
    y_top = 0.72
    box_h = 0.10
    box_w = 0.17

    # WSI Slide
    draw_rounded_box(ax, 0.03, y_top, box_w, box_h,
                     None, COLORS['dark_blue'], fontsize=11,
                     text_lines=['Whole Slide Image', '(H&E Stained)', '~100,000 x 100,000 px'])

    draw_arrow(ax, 0.03 + box_w + 0.005, y_top + box_h / 2,
               0.25 - 0.005, y_top + box_h / 2, COLORS['gray'])

    # Patch Extraction
    draw_rounded_box(ax, 0.25, y_top, box_w, box_h,
                     None, COLORS['blue'], fontsize=11,
                     text_lines=['Patch Extraction', '256 x 256 px', '@ 20x magnification'])

    draw_arrow(ax, 0.25 + box_w + 0.005, y_top + box_h / 2,
               0.47 - 0.005, y_top + box_h / 2, COLORS['gray'])

    # Foundation Model
    draw_rounded_box(ax, 0.47, y_top, box_w, box_h,
                     None, COLORS['green'], fontsize=11,
                     text_lines=['UNI2 Foundation', 'Model (ViT-L)', '1,536-dim embeddings'])

    draw_arrow(ax, 0.47 + box_w + 0.005, y_top + box_h / 2,
               0.69 - 0.005, y_top + box_h / 2, COLORS['gray'])

    # H5 Files
    draw_rounded_box(ax, 0.69, y_top, 0.14, box_h,
                     None, COLORS['teal'], fontsize=11,
                     text_lines=['H5 Files', 'N patches x 1536', 'per slide'])

    # Annotation: patches count
    ax.annotate('1,000 - 30,000\npatches/slide', xy=(0.335, y_top - 0.01),
                fontsize=8, ha='center', va='top', color=COLORS['gray'],
                style='italic', family=FONT_FAMILY)

    # === Arrow down from H5 to Aggregation ===
    draw_arrow(ax, 0.76, y_top - 0.005, 0.76, 0.58 + 0.005, COLORS['orange'], linewidth=2.5)

    # Clinical data box (right side)
    draw_rounded_box(ax, 0.85, y_top - 0.02, 0.13, 0.08,
                     None, COLORS['purple'], fontsize=9,
                     text_lines=['TCGA Clinical', 'Data (Parquet)', 'Survival + Labels'])

    # === Row 2: Aggregation ===
    y_mid = 0.48
    agg_w = 0.90
    draw_rounded_box(ax, 0.05, y_mid, agg_w, 0.10,
                     None, COLORS['orange'], fontsize=12,
                     text_lines=['43 Aggregation Methods',
                                 'Central Tendency | Dispersion | Percentiles | Extreme Values | Weighted | Energy | Mega-Vectors',
                                 'Patch-level (N x 1536)  -->  Slide-level (1 x D)   where D = 1,536 to 10,752'])

    # === Arrow down to modeling ===
    draw_arrow(ax, 0.50, y_mid - 0.005, 0.50, 0.36 + 0.005, COLORS['red'], linewidth=2.5)

    # Clinical arrow into modeling
    draw_arrow(ax, 0.91, y_top - 0.02 - 0.005, 0.82, 0.36 + 0.005, COLORS['purple'], linewidth=1.5)

    # === Row 3: Three modeling branches ===
    y_bot = 0.22
    branch_w = 0.26
    branch_h = 0.14

    # Classification
    draw_rounded_box(ax, 0.05, y_bot, branch_w, branch_h,
                     None, COLORS['red'], fontsize=11,
                     text_lines=['Classification', '24 Models', '(Tumor vs Normal)',
                                 'AUC | F1 | Accuracy'])

    # Survival
    draw_rounded_box(ax, 0.37, y_bot, branch_w, branch_h,
                     None, COLORS['dark_green'], fontsize=11,
                     text_lines=['Survival Analysis', '16 Models', '(Prognostic Prediction)',
                                 'C-index'])

    # Clustering
    draw_rounded_box(ax, 0.69, y_bot, branch_w, branch_h,
                     None, COLORS['purple'], fontsize=11,
                     text_lines=['Clustering Analysis', 'K-means + Hierarchical', '(Subtype Discovery)',
                                 'Silhouette | Log-rank'])

    # Arrows from aggregation to three branches
    draw_arrow(ax, 0.25, y_mid - 0.005, 0.18, y_bot + branch_h + 0.005, COLORS['gray'])
    draw_arrow(ax, 0.50, y_mid - 0.005, 0.50, y_bot + branch_h + 0.005, COLORS['gray'])
    draw_arrow(ax, 0.75, y_mid - 0.005, 0.82, y_bot + branch_h + 0.005, COLORS['gray'])

    # === Row 4: Results ===
    y_res = 0.06
    draw_rounded_box(ax, 0.15, y_res, 0.70, 0.10,
                     None, COLORS['dark_gray'], fontsize=12,
                     text_lines=['Results: Summary Report + Heatmaps + FDR-corrected Statistical Tests',
                                 '43 methods x 40+ models = 1,700+ experiments benchmarked'])

    # Arrows from branches to results
    draw_arrow(ax, 0.18, y_bot - 0.005, 0.35, y_res + 0.10 + 0.005, COLORS['gray'], linewidth=1.5)
    draw_arrow(ax, 0.50, y_bot - 0.005, 0.50, y_res + 0.10 + 0.005, COLORS['gray'], linewidth=1.5)
    draw_arrow(ax, 0.82, y_bot - 0.005, 0.65, y_res + 0.10 + 0.005, COLORS['gray'], linewidth=1.5)

    # Cross-validation annotation
    ax.text(0.03, 0.40, 'Stratified\n5-Fold CV',
            ha='center', va='center', fontsize=9, color=COLORS['gray'],
            style='italic', family=FONT_FAMILY,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_gray'],
                      edgecolor=COLORS['gray'], alpha=0.7))

    plt.tight_layout(pad=0.5)
    out_path = output_dir / 'fig1_pipeline_overview.png'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# FIGURE 2: Aggregation Methods Taxonomy
# =============================================================================
def create_fig2_aggregation_methods(output_dir: Path):
    """Taxonomy of all 43 aggregation methods organized by category."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.97, '43 Patch Aggregation Methods',
            ha='center', va='top', fontsize=22, fontweight='bold',
            color=COLORS['dark_gray'], family=FONT_FAMILY)
    ax.text(0.5, 0.93, 'Converting variable-length patch embeddings (N x 1,536) to fixed slide-level representations',
            ha='center', va='top', fontsize=12, color=COLORS['gray'],
            family=FONT_FAMILY)

    # Categories with their methods
    categories = [
        {
            'name': 'Central Tendency',
            'count': 6,
            'color': COLORS['blue'],
            'light': COLORS['light_blue'],
            'methods': ['mean', 'median', 'trimmed_mean', 'winsorized_mean',
                        'geometric_mean', 'harmonic_mean'],
            'dim': '1,536'
        },
        {
            'name': 'Dispersion',
            'count': 7,
            'color': COLORS['green'],
            'light': COLORS['light_green'],
            'methods': ['std', 'variance', 'range', 'coeff_of_variation',
                        'IQR', 'MAD', 'mid_range'],
            'dim': '1,536'
        },
        {
            'name': 'Distribution Shape',
            'count': 3,
            'color': COLORS['teal'],
            'light': COLORS['light_teal'],
            'methods': ['entropy', 'skewness', 'kurtosis'],
            'dim': '1,536'
        },
        {
            'name': 'Percentiles',
            'count': 7,
            'color': COLORS['orange'],
            'light': COLORS['light_orange'],
            'methods': ['P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95'],
            'dim': '1,536'
        },
        {
            'name': 'Extreme Values',
            'count': 8,
            'color': COLORS['red'],
            'light': COLORS['light_red'],
            'methods': ['max', 'min', 'top5_mean', 'top10_mean',
                        'bottom5_mean', 'bottom10_mean', 'top5_L1', 'median_patch'],
            'dim': '1,536'
        },
        {
            'name': 'Weighted',
            'count': 2,
            'color': COLORS['purple'],
            'light': COLORS['light_purple'],
            'methods': ['entropy_weighted', 'variance_weighted'],
            'dim': '1,536'
        },
        {
            'name': 'Energy / Magnitude',
            'count': 2,
            'color': COLORS['dark_green'],
            'light': COLORS['light_green'],
            'methods': ['RMS', 'sum_absolute_values'],
            'dim': '1,536'
        },
        {
            'name': 'Mega-Vectors',
            'count': 8,
            'color': COLORS['dark_blue'],
            'light': COLORS['light_blue'],
            'methods': ['mean_minmax (4,608d)', 'mean_extremes (7,680d)',
                        'mean_percentiles (7,680d)', 'central_spread (7,680d)',
                        'robust (7,680d)', 'full (10,752d)',
                        'distribution (6,144d)', 'all_percentiles (10,752d)'],
            'dim': '4,608 - 10,752'
        },
    ]

    # Layout: 2 columns x 4 rows
    n_cols = 2
    n_rows = 4
    margin_x = 0.04
    margin_y = 0.08
    gap_x = 0.03
    gap_y = 0.02
    box_w = (1.0 - 2 * margin_x - gap_x) / n_cols
    box_h = (0.87 - margin_y - n_rows * gap_y) / n_rows

    for i, cat in enumerate(categories):
        col = i % n_cols
        row = i // n_cols
        x = margin_x + col * (box_w + gap_x)
        y = 0.87 - (row + 1) * (box_h + gap_y)

        # Background box
        bg = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.01",
            facecolor=cat['light'], edgecolor=cat['color'],
            linewidth=2, alpha=0.85, zorder=2
        )
        ax.add_patch(bg)

        # Category header
        header_h = 0.035
        header = FancyBboxPatch(
            (x + 0.005, y + box_h - header_h - 0.005), box_w - 0.01, header_h,
            boxstyle="round,pad=0.005",
            facecolor=cat['color'], edgecolor='none', alpha=0.95, zorder=3
        )
        ax.add_patch(header)
        ax.text(x + box_w / 2, y + box_h - header_h / 2 - 0.005,
                f"{cat['name']}  ({cat['count']} methods)  |  Output: {cat['dim']} dims",
                ha='center', va='center', fontsize=9.5, fontweight='bold',
                color='white', zorder=4, family=FONT_FAMILY)

        # Method list
        methods_text = '  |  '.join(cat['methods'])
        # Wrap long text
        max_chars_per_line = 55
        lines = []
        current_line = ''
        for part in cat['methods']:
            if current_line:
                test = current_line + '  |  ' + part
            else:
                test = part
            if len(test) > max_chars_per_line and current_line:
                lines.append(current_line)
                current_line = part
            else:
                current_line = test
        if current_line:
            lines.append(current_line)

        text_y = y + box_h - header_h - 0.020
        for j, line in enumerate(lines):
            ax.text(x + box_w / 2, text_y - j * 0.022, line,
                    ha='center', va='center', fontsize=8.5,
                    color=COLORS['dark_gray'], zorder=4, family=FONT_FAMILY)

    # Summary box at bottom
    summary_y = 0.02
    draw_rounded_box(ax, 0.15, summary_y, 0.70, 0.06,
                     None, COLORS['dark_gray'], fontsize=10,
                     text_lines=['35 standard methods (1,536 dims)  +  8 mega-vectors (4,608 - 10,752 dims)',
                                 'Each method applied independently to every slide in the dataset'])

    plt.tight_layout(pad=0.5)
    out_path = output_dir / 'fig2_aggregation_methods.png'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# FIGURE 3: Modeling Pipeline Detail
# =============================================================================
def create_fig3_modeling_pipeline(output_dir: Path):
    """Classification, survival, and clustering branches in detail."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 10))

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    fig.suptitle('Modeling Pipeline: 40+ Models Across 3 Analysis Branches',
                 fontsize=18, fontweight='bold', color=COLORS['dark_gray'],
                 family=FONT_FAMILY, y=0.98)

    # ========== Panel 1: Classification ==========
    ax = axes[0]
    ax.set_title('Classification\n(Tumor vs Normal)', fontsize=13,
                 fontweight='bold', color=COLORS['red'], family=FONT_FAMILY, pad=10)

    model_groups = [
        ('Tree Ensembles (7)', COLORS['red'], COLORS['light_red'],
         ['RandomForest', 'ExtraTrees', 'GradientBoosting',
          'HistGradientBoosting', 'AdaBoost', 'Bagging', 'DecisionTree']),
        ('Boosting (3)', COLORS['orange'], COLORS['light_orange'],
         ['XGBoost*', 'LightGBM*', 'CatBoost*']),
        ('Linear (5)', COLORS['blue'], COLORS['light_blue'],
         ['LogReg-L2', 'LogReg-L1', 'LogReg-ElasticNet',
          'RidgeClassifier', 'SGDClassifier']),
        ('SVM (4)', COLORS['purple'], COLORS['light_purple'],
         ['LinearSVC', 'SVC-linear', 'SVC-RBF', 'NuSVC']),
        ('Other (5)', COLORS['teal'], COLORS['light_teal'],
         ['KNN', 'NearestCentroid', 'GaussianNB', 'BernoulliNB', 'LDA']),
    ]

    y_pos = 0.90
    for group_name, color, light, models in model_groups:
        # Group header
        header = FancyBboxPatch(
            (0.05, y_pos - 0.03), 0.90, 0.035,
            boxstyle="round,pad=0.005",
            facecolor=color, edgecolor='none', alpha=0.9, zorder=3
        )
        ax.add_patch(header)
        ax.text(0.50, y_pos - 0.012, group_name,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', zorder=4, family=FONT_FAMILY)

        # Models list
        y_pos -= 0.04
        models_str = ', '.join(models)
        ax.text(0.50, y_pos - 0.015, models_str,
                ha='center', va='center', fontsize=7.5,
                color=COLORS['dark_gray'], zorder=4, family=FONT_FAMILY,
                wrap=True)
        y_pos -= 0.04

    # Metrics box
    draw_rounded_box(ax, 0.05, 0.08, 0.90, 0.08,
                     None, COLORS['dark_gray'], fontsize=9,
                     text_lines=['Metrics', 'AUC-ROC | F1 | Accuracy | Precision | Recall'])

    # CV annotation
    ax.text(0.50, 0.02, '* = optional (graceful fallback)',
            ha='center', fontsize=7, color=COLORS['gray'],
            style='italic', family=FONT_FAMILY)

    # ========== Panel 2: Survival ==========
    ax = axes[1]
    ax.set_title('Survival Analysis\n(Prognostic Prediction)', fontsize=13,
                 fontweight='bold', color=COLORS['dark_green'], family=FONT_FAMILY, pad=10)

    surv_groups = [
        ('Cox-based (4)', COLORS['dark_green'], COLORS['light_green'],
         ['CoxPH', 'Coxnet-Lasso', 'Coxnet-Ridge', 'Coxnet-ElasticNet']),
        ('Gradient Boosting (2)', COLORS['green'], COLORS['light_green'],
         ['GradientBoostingSurvival', 'ComponentwiseGB']),
        ('Forest (2)', COLORS['teal'], COLORS['light_teal'],
         ['RandomSurvivalForest', 'ExtraSurvivalTrees']),
        ('SVM (2)', COLORS['blue'], COLORS['light_blue'],
         ['FastSurvivalSVM', 'FastKernelSurvivalSVM']),
        ('Ridge (1)', COLORS['purple'], COLORS['light_purple'],
         ['IPCRidge']),
        ('AFT Models (4)', COLORS['orange'], COLORS['light_orange'],
         ['CoxPHFitter', 'WeibullAFT', 'LogNormalAFT', 'LogLogisticAFT']),
    ]

    y_pos = 0.90
    for group_name, color, light, models in surv_groups:
        header = FancyBboxPatch(
            (0.05, y_pos - 0.03), 0.90, 0.035,
            boxstyle="round,pad=0.005",
            facecolor=color, edgecolor='none', alpha=0.9, zorder=3
        )
        ax.add_patch(header)
        ax.text(0.50, y_pos - 0.012, group_name,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', zorder=4, family=FONT_FAMILY)

        y_pos -= 0.04
        models_str = ', '.join(models)
        ax.text(0.50, y_pos - 0.015, models_str,
                ha='center', va='center', fontsize=7.5,
                color=COLORS['dark_gray'], zorder=4, family=FONT_FAMILY)
        y_pos -= 0.04

    # Metrics box
    draw_rounded_box(ax, 0.05, 0.08, 0.90, 0.08,
                     None, COLORS['dark_gray'], fontsize=9,
                     text_lines=['Metric', 'Concordance Index (C-index)'])

    ax.text(0.50, 0.02, 'scikit-survival + lifelines',
            ha='center', fontsize=7, color=COLORS['gray'],
            style='italic', family=FONT_FAMILY)

    # ========== Panel 3: Clustering ==========
    ax = axes[2]
    ax.set_title('Clustering & Statistics\n(Subtype Discovery)', fontsize=13,
                 fontweight='bold', color=COLORS['purple'], family=FONT_FAMILY, pad=10)

    # Clustering pipeline (vertical flow)
    steps = [
        ('PCA Reduction', '95% variance retained\nStandardScaler normalization', COLORS['blue']),
        ('K-means Clustering', 'k = 2 to 10\nOptimal k by silhouette score', COLORS['green']),
        ('Hierarchical (Ward)', 'Agglomerative clustering\nDendrogram analysis', COLORS['teal']),
        ('Chi-square Test', 'Cluster labels vs\nTumor/Normal status', COLORS['orange']),
        ('Log-rank Test', 'Cluster labels vs\nSurvival outcomes', COLORS['red']),
        ('FDR Correction', 'Benjamini-Hochberg\nalpha = 0.05', COLORS['purple']),
    ]

    step_h = 0.10
    gap = 0.025
    start_y = 0.87
    for i, (name, desc, color) in enumerate(steps):
        y = start_y - i * (step_h + gap)
        draw_rounded_box(ax, 0.08, y - step_h, 0.84, step_h,
                         None, color, fontsize=10,
                         text_lines=[name, desc])
        if i < len(steps) - 1:
            draw_arrow(ax, 0.50, y - step_h - 0.002,
                       0.50, y - step_h - gap + 0.002, COLORS['gray'])

    plt.tight_layout(pad=1.5)
    out_path = output_dir / 'fig3_modeling_pipeline.png'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# FIGURE 4: Results & Outputs
# =============================================================================
def create_fig4_results_outputs(output_dir: Path):
    """Output file structure and example performance heatmap."""
    fig = plt.figure(figsize=(16, 10))

    # Title
    fig.suptitle('Pipeline Outputs & Results Interpretation',
                 fontsize=20, fontweight='bold', color=COLORS['dark_gray'],
                 family=FONT_FAMILY, y=0.97)

    # === Left panel: File structure ===
    ax1 = fig.add_axes([0.03, 0.05, 0.35, 0.85])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    ax1.text(0.5, 0.97, 'Output File Structure', ha='center', va='top',
             fontsize=14, fontweight='bold', color=COLORS['dark_blue'],
             family=FONT_FAMILY)

    files = [
        ('results/', '', COLORS['dark_gray'], True),
        ('  aggregated_embeddings/', '43 CSV files', COLORS['blue'], True),
        ('    mean.csv', '(110 x 1536)', COLORS['gray'], False),
        ('    mega_full.csv', '(110 x 10752)', COLORS['gray'], False),
        ('    ...', '', COLORS['gray'], False),
        ('  classification_results/', '', COLORS['red'], True),
        ('    all_results.csv', '1,032 rows', COLORS['gray'], False),
        ('  survival_results/', '', COLORS['dark_green'], True),
        ('    all_results.csv', '688 rows', COLORS['gray'], False),
        ('  clustering_results/', '', COLORS['purple'], True),
        ('    all_results.json', 'per-method', COLORS['gray'], False),
        ('  statistical_tests/', '', COLORS['orange'], True),
        ('    fdr_corrected.csv', 'p-values', COLORS['gray'], False),
        ('  figures/', '', COLORS['teal'], True),
        ('    classification_heatmap.png', '', COLORS['gray'], False),
        ('    survival_heatmap.png', '', COLORS['gray'], False),
        ('    pca_variance.png', '', COLORS['gray'], False),
        ('  summary_report.csv', 'Best per method', COLORS['dark_gray'], True),
    ]

    y = 0.91
    for name, desc, color, is_dir in files:
        weight = 'bold' if is_dir else 'normal'
        size = 9 if is_dir else 8
        ax1.text(0.05, y, name, ha='left', va='center',
                 fontsize=size, fontweight=weight, color=color,
                 family='DejaVu Sans Mono')
        if desc:
            ax1.text(0.75, y, desc, ha='left', va='center',
                     fontsize=7.5, color=COLORS['gray'],
                     family=FONT_FAMILY, style='italic')
        y -= 0.045

    # === Right panel: Example heatmap ===
    ax2 = fig.add_axes([0.42, 0.35, 0.55, 0.55])

    ax2.set_title('Example: Classification AUC Heatmap\n(43 aggregation methods x 24 models)',
                  fontsize=11, fontweight='bold', color=COLORS['dark_gray'],
                  family=FONT_FAMILY, pad=10)

    # Generate synthetic heatmap data
    np.random.seed(42)
    n_methods = 15  # Show subset for readability
    n_models = 12

    method_names = ['mean', 'median', 'trimmed_mean', 'std', 'P75', 'P90',
                    'top5_mean', 'entropy_wt', 'RMS', 'mega_full',
                    'mega_robust', 'variance_wt', 'IQR', 'max', 'kurtosis']
    model_names = ['RF', 'XGB', 'LightGBM', 'HistGB', 'LogReg', 'SVC-RBF',
                   'ExtraTrees', 'AdaBoost', 'KNN', 'LDA', 'GaussNB', 'Ridge']

    # Realistic-looking AUC values (0.5-1.0 range)
    base = 0.70
    data = base + np.random.beta(3, 2, (n_methods, n_models)) * 0.25
    # Make mega-vectors generally better
    data[9:12, :] += 0.03
    data = np.clip(data, 0.5, 1.0)

    sns.heatmap(data, ax=ax2, cmap='RdYlGn', vmin=0.5, vmax=1.0,
                xticklabels=model_names, yticklabels=method_names,
                annot=True, fmt='.2f', annot_kws={'size': 7},
                cbar_kws={'label': 'AUC-ROC', 'shrink': 0.8})

    ax2.set_xlabel('Classification Models', fontsize=10, family=FONT_FAMILY)
    ax2.set_ylabel('Aggregation Methods', fontsize=10, family=FONT_FAMILY)
    ax2.tick_params(labelsize=8)

    # Note
    ax2.text(0.5, -0.12, '(Illustrative example with simulated values — actual results depend on data)',
             ha='center', fontsize=8, color=COLORS['gray'], style='italic',
             transform=ax2.transAxes, family=FONT_FAMILY)

    # === Bottom right: Key numbers ===
    ax3 = fig.add_axes([0.42, 0.05, 0.55, 0.22])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    ax3.text(0.5, 0.95, 'Pipeline Scale', ha='center', va='top',
             fontsize=13, fontweight='bold', color=COLORS['dark_gray'],
             family=FONT_FAMILY)

    stats_items = [
        ('43', 'Aggregation\nMethods', COLORS['orange']),
        ('24', 'Classification\nModels', COLORS['red']),
        ('16', 'Survival\nModels', COLORS['dark_green']),
        ('1,700+', 'Total\nExperiments', COLORS['dark_blue']),
    ]

    for i, (number, label, color) in enumerate(stats_items):
        x = 0.12 + i * 0.22
        # Number circle
        circle = plt.Circle((x, 0.55), 0.08, facecolor=color, edgecolor='white',
                             linewidth=2, alpha=0.9, zorder=3, transform=ax3.transData)
        ax3.add_patch(circle)
        ax3.text(x, 0.55, number, ha='center', va='center',
                 fontsize=14, fontweight='bold', color='white',
                 zorder=4, family=FONT_FAMILY)
        ax3.text(x, 0.25, label, ha='center', va='center',
                 fontsize=9, color=COLORS['dark_gray'],
                 family=FONT_FAMILY)

    out_path = output_dir / 'fig4_results_outputs.png'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate WSI pipeline workflow figures')
    parser.add_argument('--output-dir', type=str, default='./figures',
                        help='Directory to save figures (default: ./figures)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating WSI Embedding Analysis Pipeline Figures")
    print("=" * 50)

    print("\n[1/4] Pipeline Overview...")
    create_fig1_pipeline_overview(output_dir)

    print("[2/4] Aggregation Methods Taxonomy...")
    create_fig2_aggregation_methods(output_dir)

    print("[3/4] Modeling Pipeline Detail...")
    create_fig3_modeling_pipeline(output_dir)

    print("[4/4] Results & Outputs...")
    create_fig4_results_outputs(output_dir)

    print(f"\nAll 4 figures saved to: {output_dir}/")


if __name__ == '__main__':
    main()
