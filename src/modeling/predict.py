"""Score the live customer base (features as of the day after the last order)."""

from __future__ import annotations

import joblib
import pandas as pd

from src import config
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RFM_COLUMNS, build_rfm_features
from src.modeling.segment import PLAYBOOK


def score_customers(
    tx: pd.DataFrame,
    customers: pd.DataFrame,
    scoring_date: str,
    segment_model,
    segment_names: dict,
    churn_model,
    high_risk_threshold: float = 0.6,
) -> pd.DataFrame:
    """Return one row per customer: RFM, segment, churn probability, risk tier.

    Both models were FITTED on the earlier training snapshot; here they are only
    applied (``transform`` / ``predict``), never refitted.
    """
    feats = build_rfm_features(tx, customers, scoring_date)
    feats["segment"] = [segment_names[c] for c in segment_model.predict(feats[RFM_COLUMNS])]
    feats["churn_probability"] = churn_model.predict_proba(
        feats[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )[:, 1]
    feats["risk_tier"] = pd.cut(
        feats["churn_probability"],
        bins=[-0.01, 0.3, high_risk_threshold, 1.0],
        labels=["Low", "Medium", "High"],
    )
    feats["recommended_action"] = feats["segment"].map(
        lambda s: PLAYBOOK.get(s.rstrip(" 0123456789"), "")
    )
    # Win-back list: still-recent customers (<=180d) with high churn risk, ranked by
    # expected revenue loss. Long-lapsed customers are 'already churned' - not actionable.
    feats["revenue_at_risk"] = (
        feats["churn_probability"]
        * feats["avg_order_value"]
        * feats["frequency"]
        / (feats["tenure_days"].clip(lower=30) / 365)
    )
    feats["is_winback_target"] = (feats["recency_days"] <= 180) & (feats["risk_tier"] == "High")
    return feats


def load_models():
    """Load the persisted segmentation pipeline, segment names and churn model."""
    return (
        joblib.load(config.MODELS_DIR / "kmeans_rfm_pipeline.joblib"),
        joblib.load(config.MODELS_DIR / "segment_names.joblib"),
        joblib.load(config.MODELS_DIR / "churn_model.joblib"),
    )
