#!/usr/bin/env python3
"""Generate plots for the Mammography analysis and save them to results/plots.

Produces:
- results/plots/univariate/<feature>.png
- results/plots/feature_importance.png (MDI / Permutation / SHAP side-by-side)
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
UNIV = PLOTS / "univariate"
PLOTS.mkdir(parents=True, exist_ok=True)
UNIV.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")

def plot_univariate(data_path):
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print("Could not read data for univariate plots:", e)
        return
    # choose numeric columns (exclude 'class')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'class' in num_cols:
        num_cols.remove('class')
    for col in num_cols:
        try:
            plt.figure(figsize=(6,3))
            sns.histplot(df[col].dropna(), kde=True, stat='density', bins=30)
            plt.title(f'Distribution: {col}')
            plt.xlabel(col)
            plt.tight_layout()
            out = UNIV / f"univariate_{col}.png"
            plt.savefig(out, dpi=150)
            plt.close()
            print('Saved', out)
        except Exception as e:
            print('Failed to plot', col, e)

def plot_importances(fi_csv):
    try:
        df = pd.read_csv(fi_csv)
    except Exception as e:
        print('Could not read feature importance CSV:', e)
        return
    # ensure columns present
    cols = df.columns.tolist()
    fig, axes = plt.subplots(1,3, figsize=(15,6))
    # MDI
    if 'mdi' in cols:
        mdi = df.sort_values('mdi', ascending=False)
        sns.barplot(x='mdi', y='feature', data=mdi.head(15), ax=axes[0])
        axes[0].set_title('MDI')
    else:
        axes[0].text(0.5,0.5,'MDI missing', ha='center')
    # Permutation
    if 'perm_mean' in cols:
        perm = df.sort_values('perm_mean', ascending=False)
        sns.barplot(x='perm_mean', y='feature', data=perm.head(15), ax=axes[1])
        axes[1].set_title('Permutation (mean)')
    else:
        axes[1].text(0.5,0.5,'Permutation missing', ha='center')
    # SHAP
    if 'shap_mean' in cols:
        shap = df.sort_values('shap_mean', ascending=False)
        sns.barplot(x='shap_mean', y='feature', data=shap.head(15), ax=axes[2])
        axes[2].set_title('SHAP mean(|value|)')
    else:
        axes[2].text(0.5,0.5,'SHAP missing', ha='center')

    plt.tight_layout()
    out = PLOTS / 'feature_importance.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print('Saved', out)

def main():
    # try primary data saved by notebook
    data_candidates = [ROOT / 'data' / 'raw' / 'Mammography_from_notebook.csv', ROOT / 'data' / 'raw' / 'Mammography.csv', ROOT / 'data' / 'raw' / 'Mammography_from_openml.csv']
    for p in data_candidates:
        if p.exists():
            plot_univariate(p)
            break
    else:
        print('No raw data CSV found for univariate plots; skipping.')

    # feature importance
    fi = RESULTS / 'feature_importance_comparison.csv'
    if fi.exists():
        plot_importances(fi)
    else:
        print('feature_importance_comparison.csv not found; skipping importance plot')

if __name__ == '__main__':
    main()
