"""Cleaning, RFM and churn-feature engineering.

Design rule (prevents data leakage): every function that builds features takes a
``snapshot`` timestamp and uses ONLY transactions strictly before it. Churn labels
are built separately from transactions on/after the snapshot.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from src import config

# --------------------------------------------------------------------------- #
# Extraction (SQL -> pandas)
# --------------------------------------------------------------------------- #


def get_engine():
    """Return a SQLAlchemy engine for the project database."""
    return create_engine(config.DATABASE_URL)


def run_sql_file(name: str) -> None:
    """Execute a DDL/view SQL file from the sql/ directory."""
    with get_engine().begin() as conn:
        conn.execute(text((config.SQL_DIR / name).read_text()))


def load_clean_view() -> pd.DataFrame:
    """Read the de-duplicated transaction view from PostgreSQL."""
    run_sql_file("02_clean_view.sql")
    engine = get_engine()
    query = "SELECT * FROM v_transactions_clean ORDER BY order_date"
    df = pd.read_sql(query, engine, parse_dates=["order_date"])
    with engine.connect() as conn:
        n_db = conn.execute(text("SELECT COUNT(*) FROM transactions_raw")).scalar()
    df.attrs["rows_in_db"] = int(n_db)  # lets the audit count SQL-side de-duplication
    return df


def extract_user_level(snapshot: str = config.SNAPSHOT_DATE) -> pd.DataFrame:
    """Run the flattened user-level SQL extract (sql/03_extract_user_level.sql)."""
    sql = (config.SQL_DIR / "03_extract_user_level.sql").read_text()
    return pd.read_sql(
        text(sql),
        get_engine(),
        params={"snapshot": snapshot},
        parse_dates=["signup_date", "first_order_date", "last_order_date"],
    )


# --------------------------------------------------------------------------- #
# Cleaning (programmatic; never drops > 5 % of rows)
# --------------------------------------------------------------------------- #


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Impute / cap instead of dropping. Returns (clean_df, audit_report).

    * missing unit_price -> median of the product category
    * missing quantity   -> 1 (mode)
    * missing channel    -> "unknown"
    * price outliers     -> winsorised at the per-category 99th percentile
    * negative quantity  -> kept, flagged as ``is_return`` (excluded from revenue)
    """
    df = raw.copy()
    n_raw = int(raw.attrs.get("rows_in_db", len(df)))
    n_dupes_removed = (n_raw - len(df)) + int(df.duplicated().sum())
    df = df.drop_duplicates()

    audit = {
        "rows_raw": n_raw,
        "exact_duplicates_removed": n_dupes_removed,
        "missing_unit_price": int(df["unit_price"].isna().sum()),
        "missing_quantity": int(df["quantity"].isna().sum()),
        "missing_channel": int(df["channel"].isna().sum()),
    }

    cat_median = df.groupby("category")["unit_price"].transform("median")
    df["unit_price"] = df["unit_price"].fillna(cat_median)
    df["quantity"] = df["quantity"].fillna(1)
    df["channel"] = df["channel"].fillna("unknown")

    cap = df.groupby("category")["unit_price"].transform(lambda s: s.quantile(0.99))
    audit["price_outliers_capped"] = int((df["unit_price"] > cap).sum())
    df["unit_price"] = df["unit_price"].clip(upper=cap)

    df["is_return"] = df["quantity"] < 0
    df["quantity"] = df["quantity"].abs()
    df["line_value"] = df["unit_price"] * df["quantity"] * (1 - df["discount_pct"])
    df["line_value"] = df["line_value"].where(~df["is_return"], 0.0)

    audit["rows_clean"] = len(df)
    audit["pct_rows_dropped"] = round(100 * (n_raw - len(df)) / n_raw, 3)
    assert audit["pct_rows_dropped"] < 5, "Data-integrity target violated (>5% dropped)"
    return df.reset_index(drop=True), audit


# --------------------------------------------------------------------------- #
# RFM + behavioural features (snapshot-safe)
# --------------------------------------------------------------------------- #


def build_rfm_features(tx: pd.DataFrame, customers: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """One row per customer with RFM and behavioural features as of ``snapshot``."""
    snap = pd.Timestamp(snapshot)
    hist = tx[tx["order_date"] < snap]

    orders = (
        hist.groupby(["customer_id", "order_id"])
        .agg(
            order_date=("order_date", "min"),
            order_value=("line_value", "sum"),
            n_lines=("line_value", "size"),
            has_return=("is_return", "max"),
            max_discount=("discount_pct", "max"),
        )
        .reset_index()
    )

    g = orders.groupby("customer_id")
    feats = pd.DataFrame(
        {
            "recency_days": (snap - g["order_date"].max()).dt.days,
            "frequency": g["order_id"].nunique(),
            "monetary": g["order_value"].sum(),
            "avg_order_value": g["order_value"].mean(),
            "avg_lines_per_order": g["n_lines"].mean(),
            "return_rate": g["has_return"].mean(),
            "discount_order_share": g["max_discount"].apply(lambda s: (s > 0).mean()),
        }
    )

    # Inter-purchase gap (needs >= 2 orders)
    orders = orders.sort_values(["customer_id", "order_date"])
    gaps = orders.groupby("customer_id")["order_date"].diff().dt.days
    feats["avg_gap_days"] = gaps.groupby(orders["customer_id"]).mean()

    # Recent-activity windows
    for days in (30, 90):
        recent = orders[orders["order_date"] >= snap - pd.Timedelta(days=days)]
        rg = recent.groupby("customer_id")
        feats[f"orders_last_{days}d"] = rg["order_id"].nunique()
        feats[f"spend_last_{days}d"] = rg["order_value"].sum()
    feats = feats.fillna({c: 0 for c in feats.columns if "last_" in c})

    feats["n_categories"] = hist.groupby("customer_id")["category"].nunique()
    top_channel = (
        hist.groupby("customer_id")["channel"].agg(lambda s: s.mode().iat[0]).rename("top_channel")
    )
    feats = feats.join(top_channel)

    cust = customers.set_index("customer_id")
    feats = feats.join(cust[["signup_date", "city", "acquisition_channel"]], how="left")
    feats["tenure_days"] = (snap - feats["signup_date"]).dt.days
    feats["gap_ratio"] = feats["recency_days"] / feats["avg_gap_days"].fillna(
        feats["tenure_days"]
    ).clip(lower=1)
    feats["avg_gap_days"] = feats["avg_gap_days"].fillna(feats["tenure_days"])
    return feats.drop(columns="signup_date").reset_index()


def build_churn_labels(
    tx: pd.DataFrame,
    snapshot: str = config.SNAPSHOT_DATE,
    window_days: int = config.CHURN_WINDOW_DAYS,
) -> pd.Series:
    """1 = customer placed NO order in [snapshot, snapshot + window_days)."""
    snap = pd.Timestamp(snapshot)
    end = snap + pd.Timedelta(days=window_days)
    future = tx[(tx["order_date"] >= snap) & (tx["order_date"] < end)]
    active = set(future["customer_id"].unique())
    return pd.Series(
        {cid: int(cid not in active) for cid in tx["customer_id"].unique()},
        name="churned",
    )


def assemble_model_table(
    tx: pd.DataFrame, customers: pd.DataFrame, snapshot: str = config.SNAPSHOT_DATE
) -> pd.DataFrame:
    """Features at ``snapshot`` joined with the forward-looking churn label."""
    feats = build_rfm_features(tx, customers, snapshot)
    labels = build_churn_labels(tx, snapshot)
    feats["churned"] = feats["customer_id"].map(labels).fillna(1).astype(int)
    return feats


# Columns the models are allowed to see (no ids, no label)
NUMERIC_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_lines_per_order",
    "return_rate",
    "discount_order_share",
    "avg_gap_days",
    "orders_last_30d",
    "spend_last_30d",
    "orders_last_90d",
    "spend_last_90d",
    "n_categories",
    "tenure_days",
    "gap_ratio",
]
CATEGORICAL_FEATURES = ["top_channel", "acquisition_channel", "city"]
RFM_COLUMNS = ["recency_days", "frequency", "monetary"]
