# MATH3310_FA_2025_P02_G03_SMITH
Group Project 2

Project layout (Cookiecutter-style, simplified):

- `data/`
	- `raw/` : original/raw data files (ARFF, raw CSV exports)
	- `processed/` : cleaned and processed datasets (derived)
- `notebooks/` : Jupyter notebooks (analysis, EDA, experiments)
- `src/` : source code (scripts, modules)
- `models/` : persisted model artifacts (joblib files, pickles)
- `results/` : experiment outputs and result CSVs (metrics, comparisons)
- `reports/` : generated reports, figures, write-ups
- `docs/` : documentation and supplementary files

Quick start (reproduce results locally)

1. Create and activate a virtual environment (Linux/macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Recompute experiments (resampling, SHAP) and generate artifacts:

```bash
# recompute models / resampling / SHAP (may take several minutes)
.venv/bin/python scripts/recompute_resampling_shap.py

# produce plots and additional artifacts
.venv/bin/python scripts/generate_plots.py
.venv/bin/python scripts/produce_additional_artifacts.py
```

Outputs are written to `results/` and `results/plots/` (feature importance images, gridsearch heatmap, metrics comparison).

Notes:
- If you plan to re-run the notebook interactively, open `notebooks/mp2.ipynb` in JupyterLab and select the `.venv` kernel.
- If `imbalanced-learn` or `shap` are not available, some experiments (SMOTE/undersampling or SHAP) will be skipped. They are included in `requirements.txt` below.