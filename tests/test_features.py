"""Unit tests for leakage safety and cleaning (run: pytest -q)."""

import pandas as pd

from src import features as F


def _tx():
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "customer_id": [1, 1, 2, 2],
            "order_date": pd.to_datetime(["2025-01-01", "2025-11-15", "2025-02-01", "2025-03-01"]),
            "category": ["A", "A", "A", "A"],
            "unit_price": [10.0, 10.0, 20.0, 5000.0],
            "quantity": [1, 1, None, -1],
            "channel": ["web", None, "web", "web"],
            "discount_pct": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_cleaning_never_drops_more_than_5pct():
    clean, audit = F.clean_transactions(_tx())
    assert audit["pct_rows_dropped"] < 5
    assert clean["quantity"].notna().all() and (clean["channel"] == "unknown").sum() == 1


def test_returns_excluded_from_revenue():
    clean, _ = F.clean_transactions(_tx())
    assert clean.loc[clean["is_return"], "line_value"].eq(0).all()


def test_features_ignore_post_snapshot_orders():
    clean, _ = F.clean_transactions(_tx())
    cust = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "signup_date": pd.to_datetime(["2024-12-01"] * 2),
            "city": ["X", "X"],
            "acquisition_channel": ["organic"] * 2,
        }
    )
    feats = F.build_rfm_features(clean, cust, "2025-10-31").set_index("customer_id")
    assert feats.loc[1, "frequency"] == 1  # the 2025-11-15 order must NOT be counted


def test_churn_label_uses_only_future_window():
    clean, _ = F.clean_transactions(_tx())
    labels = F.build_churn_labels(clean, "2025-10-31", 60)
    assert labels[1] == 0 and labels[2] == 1
