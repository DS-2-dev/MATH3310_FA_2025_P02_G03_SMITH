#!/usr/bin/env python3
"""Recompute resampling experiments and feature importances for the Mammography dataset.

This script fetches the Mammography dataset from OpenML, runs baseline, weighted,
SMOTE and RandomUnderSampler experiments (inside CV), computes permutation and
SHAP importances (when possible), and writes results to the `results/` folder.

Run inside the repository venv: `.venv/bin/python scripts/recompute_resampling_shap.py`
"""
import warnings
warnings.filterwarnings("ignore")

import os
import random

# Make results reproducible/deterministic where possible
SEED = 42
# PYTHONHASHSEED controls hash randomization
os.environ.setdefault('PYTHONHASHSEED', str(SEED))
random.seed(SEED)

from pathlib import Path
import numpy as np
import pandas as pd
# numpy RNG seed
np.random.seed(SEED)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def gmean_from_confmat(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return np.sqrt(sens * spec)


def main():
    print("Fetching Mammography dataset from OpenML...")
    data = fetch_openml(name="mammography", version=1, as_frame=True)
    df = data.frame.copy()

    # Determine target column and normalize to 0/1
    target_col = data.target.name if hasattr(data, "target") else "class"
    if df[target_col].dtype.kind in "OU":
        classes = sorted(df[target_col].unique())
        mapping = {classes[0]: 0, classes[1]: 1}
        df[target_col] = df[target_col].map(mapping)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X_train.select_dtypes(include=[object, "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ]
    )

    # Use single-threaded training/prediction to reduce nondeterminism from parallelism
    base_clf = RandomForestClassifier(random_state=SEED, n_jobs=1)
    baseline_pipe = Pipeline([("preproc", preprocessor), ("clf", base_clf)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline CV
    print("Running baseline CV (ROC AUC)...")
    baseline_aucs = cross_val_score(baseline_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    print("Baseline ROC AUC mean:", baseline_aucs.mean())

    # Weighted RF variants
    print("Evaluating weighted RandomForest variants...")
    weighted_rows = []
    for name, cw in [("unweighted", None), ("balanced", "balanced"), ("balanced_subsample", "balanced_subsample")]:
        clf = RandomForestClassifier(class_weight=cw, random_state=SEED, n_jobs=1)
        pipe = Pipeline([("preproc", preprocessor), ("clf", clf)])
        aucs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        preds = cross_val_predict(pipe, X_train, y_train, cv=cv, n_jobs=1)
        from sklearn.metrics import precision_score, recall_score
        weighted_rows.append({
            "variant": name,
            "class_weight": cw,
            "accuracy_mean": accuracy_score(y_train, preds),
            "accuracy_std": 0.0,
            "roc_auc_mean": aucs.mean(),
            "roc_auc_std": aucs.std(),
            "f1_mean": f1_score(y_train, preds, average="binary"),
            "f1_std": 0.0,
            "precision_mean": precision_score(y_train, preds, zero_division=0),
            "precision_std": 0.0,
            "recall_mean": recall_score(y_train, preds, zero_division=0),
            "recall_std": 0.0
        })

    pd.DataFrame(weighted_rows).to_csv(RESULTS_DIR / "weighted_rf_cv_results.csv", index=False)
    print("Saved weighted_rf_cv_results.csv")

    # Resampling experiments
    print("Running resampling experiments (none, SMOTE, RandomUnderSampler) ...")
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline as ImbPipeline
        smote_available = True
    except Exception:
        print("imbalanced-learn not available; will skip SMOTE/RUS")
        smote_available = False

    res_rows = []
    # baseline
    aucs = cross_val_score(baseline_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    preds = cross_val_predict(baseline_pipe, X_train, y_train, cv=cv, n_jobs=1)
    from sklearn.metrics import precision_score, recall_score
    res_rows.append({
        "resampling": "none",
        "roc_auc_mean": aucs.mean(),
        "roc_auc_std": aucs.std(),
        "f1_mean": f1_score(y_train, preds, average="binary"),
        "f1_std": 0.0,
        "precision_mean": precision_score(y_train, preds, zero_division=0),
        "precision_std": 0.0,
        "recall_mean": recall_score(y_train, preds, zero_division=0),
        "recall_std": 0.0
    })

    if smote_available:
        smote_pipe = ImbPipeline([("preproc", preprocessor), ("smote", SMOTE(random_state=SEED)), ("clf", RandomForestClassifier(random_state=SEED, n_jobs=1))])
        rus_pipe = ImbPipeline([("preproc", preprocessor), ("rus", RandomUnderSampler(random_state=SEED)), ("clf", RandomForestClassifier(random_state=SEED, n_jobs=1))])
        for name, p in [("SMOTE", smote_pipe), ("RandomUnderSampler", rus_pipe)]:
            aucs = cross_val_score(p, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
            preds = cross_val_predict(p, X_train, y_train, cv=cv, n_jobs=1)
            res_rows.append({
                "resampling": name,
                "roc_auc_mean": aucs.mean(),
                "roc_auc_std": aucs.std(),
                "f1_mean": f1_score(y_train, preds, average="binary"),
                "f1_std": 0.0,
                "precision_mean": precision_score(y_train, preds, zero_division=0),
                "precision_std": 0.0,
                "recall_mean": recall_score(y_train, preds, zero_division=0),
                "recall_std": 0.0
            })

    pd.DataFrame(res_rows).to_csv(RESULTS_DIR / "resampling_rf_cv_results.csv", index=False)
    print("Saved resampling_rf_cv_results.csv")

    # Detailed metrics
    print("Computing detailed metrics (error, f1, gmean) for baseline RF...")
    preds = cross_val_predict(baseline_pipe, X_train, y_train, cv=cv, n_jobs=1)
    err = 1 - accuracy_score(y_train, preds)
    f1 = f1_score(y_train, preds, average="binary")
    g = gmean_from_confmat(y_train, preds)
    pd.DataFrame([{
        "resampling": "none",
        "error_rate_mean": err,
        "error_rate_std": 0.0,
        "f1_mean": f1,
        "f1_std": 0.0,
        "gmean_mean": g,
        "gmean_std": 0.0
    }]).to_csv(RESULTS_DIR / "detailed_metrics_rf_cv_results.csv", index=False)
    print("Saved detailed_metrics_rf_cv_results.csv")

    # Fit baseline pipeline on training set to compute importances
    print("Fitting baseline pipeline on full training set for importances...")
    baseline_pipe.fit(X_train, y_train)

    # Feature names after preprocessing
    try:
        feat_names = []
        feat_names.extend(num_features)
        if cat_features:
            o = preprocessor.named_transformers_["cat"]
            try:
                onehot_names = list(o.get_feature_names_out(cat_features))
            except Exception:
                onehot_names = []
            feat_names.extend(onehot_names)
    except Exception:
        feat_names = [f"f{i}" for i in range(baseline_pipe.named_steps["clf"].n_features_)]

    mdi = baseline_pipe.named_steps["clf"].feature_importances_
    # permutation_importance: set random_state and single-threaded for reproducibility
    perm = permutation_importance(baseline_pipe, X_test, y_test, n_repeats=20, random_state=SEED, n_jobs=1, scoring="roc_auc")

    fi_df = pd.DataFrame({
        "feature": feat_names,
        "mdi": mdi,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std
    })

    # SHAP: try shap.Explainer with pipeline, fallback to TreeExplainer on raw classifier
    try:
        import shap
        print("Attempting SHAP computation (this can be slow)...")
        vals = None
        try:
            explainer = shap.Explainer(baseline_pipe, X_train)
            shap_exp = explainer(X_test)
            vals = getattr(shap_exp, "values", None)
        except Exception:
            # fallback: use TreeExplainer on the fitted classifier with transformed data
            model = baseline_pipe.named_steps["clf"]
            explainer = shap.TreeExplainer(model)
            X_test_trans = preprocessor.transform(X_test)
            vals = explainer.shap_values(X_test_trans)
        arr = np.asarray(vals)
        # normalize arr into per-sample x per-feature
        if arr.ndim == 1:
            shap_mean = np.abs(arr)
        elif arr.ndim == 2:
            # (n_samples, n_features) -> mean over samples
            shap_mean = np.abs(arr).mean(axis=0)
        elif arr.ndim == 3:
            # Could be (n_samples, n_features, n_classes) OR (n_samples, n_classes, n_features)
            n0, n1, n2 = arr.shape
            if n1 == len(feat_names):
                # arr: (n_samples, n_features, n_classes) -> choose positive class on last axis
                if n2 >= 2:
                    shap_mean = np.abs(arr[:, :, 1]).mean(axis=0)
                else:
                    shap_mean = np.abs(arr).mean(axis=(0, 1))
            elif n2 == len(feat_names):
                # arr: (n_samples, n_classes, n_features) -> choose positive class on axis 1
                if n1 >= 2:
                    shap_mean = np.abs(arr[:, 1, :]).mean(axis=0)
                else:
                    shap_mean = np.abs(arr).mean(axis=(0, 1))
            else:
                # fallback: average over first two axes
                shap_mean = np.abs(arr).mean(axis=(0, 1))
        else:
            raise ValueError(f"Unexpected SHAP array ndim={arr.ndim}")

        shap_mean = np.asarray(shap_mean)
        if shap_mean.size != len(feat_names):
            raise ValueError(f"SHAP feature length ({shap_mean.size}) does not match feature names ({len(feat_names)})")
        fi_df["shap_mean"] = shap_mean
        print("SHAP computed and added to feature importance DataFrame")
    except Exception as e:
        print("SHAP not available or failed:", e)

    fi_df.sort_values(by="mdi", ascending=False).to_csv(RESULTS_DIR / "feature_importance_comparison.csv", index=False)
    print("Saved feature_importance_comparison.csv")


if __name__ == "__main__":
    main()

