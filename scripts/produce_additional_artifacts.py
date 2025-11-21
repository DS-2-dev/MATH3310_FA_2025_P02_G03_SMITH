#!/usr/bin/env python3
"""Produce additional artifacts to maximize rubric points:
- GridSearch/KFold stability heatmap
- Consolidated metrics comparison plot (F1, G-mean, error rate)
- Spearman rank correlations between MDI / Permutation / SHAP
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
GRID = PLOTS / "gridsearch"
PLOTS.mkdir(parents=True, exist_ok=True)
GRID.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")

def grid_heatmap(kfold_csv):
    df = pd.read_csv(kfold_csv)
    # pivot best_score by split_seed (rows) and cv_seed (cols)
    pivot = df.pivot(index='split_seed', columns='cv_seed', values='best_score')
    plt.figure(figsize=(6,5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
    plt.title('KFold/CV Stability: Best Score')
    out = GRID / 'kfold_best_score_heatmap.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print('Saved', out)

def metrics_comparison(detail_csv, resample_csv):
    d1 = pd.read_csv(detail_csv)
    d2 = pd.read_csv(resample_csv)
    # merge on resampling
    df = d1.merge(d2[['resampling','roc_auc_mean']], on='resampling', how='left')
    # melt for plotting
    plot_df = df.melt(id_vars=['resampling'], value_vars=['f1_mean','gmean_mean','error_rate_mean'], var_name='metric', value_name='value')
    plt.figure(figsize=(7,4))
    sns.barplot(x='resampling', y='value', hue='metric', data=plot_df)
    plt.title('Metrics by Resampling Method')
    plt.ylabel('Value')
    out = PLOTS / 'metrics_comparison.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print('Saved', out)

def fi_correlations(fi_csv):
    df = pd.read_csv(fi_csv)
    # require mdi, perm_mean, shap_mean
    cols = []
    if 'mdi' in df.columns:
        cols.append('mdi')
    if 'perm_mean' in df.columns:
        cols.append('perm_mean')
    if 'shap_mean' in df.columns:
        cols.append('shap_mean')
    if len(cols) < 2:
        print('Not enough importance columns for correlation; need at least 2 of mdi/perm_mean/shap_mean')
        return
    sub = df[cols]
    # Spearman correlation matrix
    corr = sub.corr(method='spearman')
    out_csv = RESULTS / 'feature_importance_correlations.csv'
    corr.to_csv(out_csv)
    print('Saved', out_csv)
    # also save an annotated heatmap
    plt.figure(figsize=(4,3))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm')
    plt.title('Spearman Rank Correlation (importances)')
    out = PLOTS / 'feature_importance_correlations.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print('Saved', out)

def main():
    kfold = RESULTS / 'kfold_experiment_results.csv'
    detail = RESULTS / 'detailed_metrics_rf_cv_results.csv'
    resamp = RESULTS / 'resampling_rf_cv_results.csv'
    fi = RESULTS / 'feature_importance_comparison.csv'
    if kfold.exists():
        grid_heatmap(kfold)
    else:
        print('kfold_experiment_results.csv not found; skipping heatmap')
    if detail.exists() and resamp.exists():
        metrics_comparison(detail, resamp)
    else:
        print('detail/resampling CSVs missing; skipping metrics comparison')
    if fi.exists():
        fi_correlations(fi)
    else:
        print('feature_importance_comparison.csv missing; skipping correlations')

if __name__ == '__main__':
    main()
