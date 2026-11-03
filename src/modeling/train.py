"""Supervised churn modelling: Logistic Regression vs Random Forest (+ SMOTE)."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SKEWED = [
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_gap_days",
    "spend_last_30d",
    "spend_last_90d",
    "recency_days",
    "tenure_days",
    "gap_ratio",
]


def _preprocessor(scale: bool) -> ColumnTransformer:
    """Imputation-free (features are already complete) encoder / scaler."""
    num_steps = [("log", FunctionTransformer(np.log1p, validate=False))] if scale else []
    if scale:
        num_steps.append(("scale", StandardScaler()))
    plain = [c for c in NUMERIC_FEATURES if c not in SKEWED]
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
    if scale:
        transformers.append(("skew", Pipeline(num_steps), SKEWED))
        transformers.append(("num", StandardScaler(), plain))
    else:
        transformers.append(("num", "passthrough", NUMERIC_FEATURES))
    return ColumnTransformer(transformers)


def build_models() -> dict:
    """Candidate models. SMOTE lives INSIDE the pipeline so it is applied only to
    training folds during cross-validation (no leakage into validation folds)."""
    rs = config.RANDOM_STATE
    return {
        "logreg_class_weight": Pipeline(
            [
                ("prep", _preprocessor(True)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "logreg_smote": ImbPipeline(
            [
                ("prep", _preprocessor(True)),
                ("smote", SMOTE(random_state=rs)),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        ),
        "rf_class_weight": Pipeline(
            [
                ("prep", _preprocessor(False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=rs,
                    ),
                ),
            ]
        ),
        "rf_smote": ImbPipeline(
            [
                ("prep", _preprocessor(False)),
                ("smote", SMOTE(random_state=rs)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=rs
                    ),
                ),
            ]
        ),
    }


def split(model_table: pd.DataFrame):
    """Stratified 80/20 split of customers."""
    X = model_table[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = model_table["churned"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    """Precision / Recall / F1 / ROC-AUC / PR-AUC on a held-out set."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def run_experiment(model_table: pd.DataFrame):
    """Cross-validate every candidate, refit on train, score on test.

    Returns (results_df, fitted_models, X_test, y_test).
    """
    X_tr, X_te, y_tr, y_te = split(model_table)
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    rows, fitted = [], {}
    for name, model in build_models().items():
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
        model.fit(X_tr, y_tr)
        metrics = evaluate(model, X_te, y_te)
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": cv_auc.mean(),
                "cv_roc_auc_std": cv_auc.std(),
                **metrics,
            }
        )
        fitted[name] = model
    results = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    return results.reset_index(drop=True), fitted, X_te, y_te


def save_model(model, name: str, metrics: dict) -> None:
    """Persist the chosen model (joblib) and its metrics (json) to models/."""
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
    (config.MODELS_DIR / f"{name}_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items()}, indent=2)
    )
