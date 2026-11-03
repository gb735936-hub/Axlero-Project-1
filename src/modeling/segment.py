"""RFM customer segmentation with K-Means."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config
from src.features import RFM_COLUMNS

# Recommended marketing action per business segment name
PLAYBOOK = {
    "Champions": "Reward and retain: VIP early access, loyalty tiers, referral "
    "incentives. Do NOT discount - they already buy at full price.",
    "Loyal Customers": "Grow basket size: cross-sell complementary categories, "
    "bundles and subscription / replenishment offers.",
    "Potential Loyalists": "Build the habit: personalised second/third-purchase "
    "journeys, onboarding emails, time-limited reorder incentives.",
    "At Risk": "Win back before they lapse: personalised 'we miss you' campaigns, "
    "targeted discounts, reminders based on their usual purchase gap.",
    "Hibernating": "Low-cost reactivation only: automated email/SMS flows, "
    "last-chance offers; suppress from paid-media spend if no response.",
}


def make_pipeline(k: int) -> Pipeline:
    """log1p -> standardise -> K-Means. Log tames the heavy right skew of F and M."""
    return Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=config.RANDOM_STATE)),
        ]
    )


def evaluate_k(rfm: pd.DataFrame, k_range=range(2, 11)) -> pd.DataFrame:
    """Elbow (inertia) and silhouette for each k."""
    rows = []
    X = rfm[RFM_COLUMNS].to_numpy()
    for k in k_range:
        pipe = make_pipeline(k).fit(X)
        Z = pipe[:-1].transform(X)
        labels = pipe.predict(X)
        rows.append(
            {
                "k": k,
                "inertia": pipe[-1].inertia_,
                "silhouette": silhouette_score(
                    Z, labels, sample_size=3000, random_state=config.RANDOM_STATE
                ),
            }
        )
    return pd.DataFrame(rows)


def name_clusters(rfm: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Map raw cluster ids to business names from centroid RFM behaviour.

    Rules (applied to per-cluster medians):
      * RECENT clusters (recency <= median of cluster recencies):
          highest frequency -> Champions; others -> Loyal Customers if their
          median frequency >= 6 else Potential Loyalists
      * STALE clusters: highest monetary value -> At Risk; the rest -> Hibernating
    Duplicate names get a numeric suffix so every segment stays distinct.
    """
    df = rfm[RFM_COLUMNS].copy()
    df["cluster"] = labels
    med = df.groupby("cluster")[RFM_COLUMNS].median()
    recent = med[med["recency_days"] <= med["recency_days"].median()]
    stale = med.drop(index=recent.index)

    names: dict[int, str] = {}
    for rank, (cid, row) in enumerate(recent.sort_values("frequency", ascending=False).iterrows()):
        if rank == 0:
            names[cid] = "Champions"
        else:
            names[cid] = "Loyal Customers" if row["frequency"] >= 6 else "Potential Loyalists"
    for rank, cid in enumerate(stale.sort_values("monetary", ascending=False).index):
        names[cid] = "At Risk" if rank == 0 else "Hibernating"

    seen: dict[str, int] = {}
    for cid in sorted(names):
        base = names[cid]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            names[cid] = f"{base} {seen[base]}"
    return names


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-segment size, RFM medians and revenue share."""
    out = (
        df.groupby("segment")
        .agg(
            customers=("customer_id", "size"),
            recency_median=("recency_days", "median"),
            frequency_median=("frequency", "median"),
            monetary_median=("monetary", "median"),
            revenue=("monetary", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )
    out["pct_customers"] = 100 * out["customers"] / out["customers"].sum()
    out["pct_revenue"] = 100 * out["revenue"] / out["revenue"].sum()
    out["recommended_action"] = [PLAYBOOK.get(s.rstrip(" 0123456789"), "") for s in out.index]
    return out.round(2)
