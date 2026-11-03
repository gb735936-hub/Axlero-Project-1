"""Data generation and database loading for the D2C retail project.

Run as a script to (re)create the immutable raw CSVs:

    python -m src.dataset
"""

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

CATEGORIES = {
    "Apparel": (25, 120),
    "Footwear": (40, 180),
    "Beauty": (10, 80),
    "Home & Living": (15, 200),
    "Electronics": (30, 400),
    "Sports": (20, 150),
}
CHANNELS = ["web", "mobile_app", "marketplace"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Kolkata"]

# Latent behavioural archetypes used ONLY to simulate data. Models never see them.
ARCHETYPES = {
    #               share  orders/yr  basket  decay (churn hazard)
    "champion": (0.12, 18.0, 95.0, 0.0004),
    "loyal": (0.23, 10.0, 60.0, 0.0010),
    "promising": (0.25, 6.0, 45.0, 0.0020),
    "at_risk": (0.20, 4.0, 55.0, 0.0040),
    "dormant": (0.20, 2.0, 35.0, 0.0080),
}


def generate_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create the customer dimension table."""
    names = list(ARCHETYPES)
    probs = [ARCHETYPES[a][0] for a in names]
    archetype = rng.choice(names, size=n, p=probs)
    signup = pd.to_datetime(config.START_DATE) + pd.to_timedelta(
        rng.integers(0, 600, size=n), unit="D"
    )
    return pd.DataFrame(
        {
            "customer_id": np.arange(10001, 10001 + n),
            "signup_date": signup,
            "city": rng.choice(CITIES, size=n),
            "acquisition_channel": rng.choice(
                ["organic", "paid_social", "referral", "email", "influencer"],
                size=n,
                p=[0.3, 0.3, 0.15, 0.15, 0.10],
            ),
            "archetype": archetype,
        }
    )


def generate_transactions(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate order lines. Customers stop buying after a random churn date."""
    end = pd.to_datetime(config.END_DATE)
    rows = []
    order_id = 500000
    for cust in customers.itertuples(index=False):
        _, orders_per_year, basket, hazard = ARCHETYPES[cust.archetype]
        tenure_days = (end - cust.signup_date).days
        if tenure_days <= 0:
            continue
        # Geometric churn date: constant daily hazard after signup
        lifetime = rng.geometric(hazard)
        active_days = min(tenure_days, lifetime)
        expected = max(1, rng.poisson(orders_per_year * active_days / 365))
        days = np.sort(rng.integers(0, max(active_days, 1), size=expected))
        for d in days:
            order_id += 1
            n_lines = rng.integers(1, 4)
            order_date = cust.signup_date + pd.Timedelta(days=int(d))
            hour = int(rng.choice(range(24), p=_hour_probs()))
            order_ts = order_date + pd.Timedelta(hours=hour, minutes=int(rng.integers(60)))
            for _ in range(n_lines):
                cat = rng.choice(list(CATEGORIES))
                lo, hi = CATEGORIES[cat]
                price = round(float(rng.uniform(lo, hi)) * (basket / 60) ** 0.5, 2)
                qty = int(rng.choice([1, 1, 1, 2, 3], p=[0.5, 0.2, 0.1, 0.1, 0.1]))
                rows.append(
                    (
                        order_id,
                        cust.customer_id,
                        order_ts,
                        cat,
                        price,
                        qty,
                        rng.choice(CHANNELS, p=[0.45, 0.4, 0.15]),
                        float(rng.choice([0, 0, 0, 0.1, 0.2], p=[0.5, 0.2, 0.1, 0.1, 0.1])),
                    )
                )
    tx = pd.DataFrame(
        rows,
        columns=[
            "order_id",
            "customer_id",
            "order_date",
            "category",
            "unit_price",
            "quantity",
            "channel",
            "discount_pct",
        ],
    )
    return tx


def _hour_probs() -> np.ndarray:
    p = np.array([1] * 6 + [2] * 3 + [4] * 4 + [3] * 3 + [5] * 4 + [6] * 3 + [2] * 1, dtype=float)
    return p / p.sum()


def inject_data_quality_issues(tx: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Make the raw data realistically messy (missing values, outliers, dupes, returns)."""
    tx = tx.copy()
    n = len(tx)
    # ~2% missing unit prices, ~1.5% missing quantity, ~2% missing channel
    tx.loc[rng.choice(n, int(0.02 * n), replace=False), "unit_price"] = np.nan
    tx.loc[rng.choice(n, int(0.015 * n), replace=False), "quantity"] = np.nan
    tx.loc[rng.choice(n, int(0.02 * n), replace=False), "channel"] = np.nan
    # ~0.4% extreme price outliers (data-entry errors, x20-x60)
    idx = rng.choice(n, int(0.004 * n), replace=False)
    tx.loc[idx, "unit_price"] = tx.loc[idx, "unit_price"] * rng.uniform(20, 60, len(idx))
    # ~1% returns recorded as negative quantity
    idx = rng.choice(n, int(0.01 * n), replace=False)
    tx.loc[idx, "quantity"] = -tx.loc[idx, "quantity"].abs()
    # ~0.5% exact duplicate rows
    dupes = tx.sample(int(0.005 * n), random_state=config.RANDOM_STATE)
    return pd.concat([tx, dupes], ignore_index=True).sort_values("order_date")


def build_raw_data() -> None:
    """Generate and persist immutable raw CSVs to data/raw/."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    customers = generate_customers(config.N_CUSTOMERS, rng)
    tx = generate_transactions(customers, rng)
    tx = inject_data_quality_issues(tx, rng)
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    customers.drop(columns="archetype").to_csv(config.DATA_RAW / "customers.csv", index=False)
    tx.to_csv(config.DATA_RAW / "transactions.csv", index=False)
    logger.info("customers=%s transactions=%s", len(customers), len(tx))
    print(f"customers.csv: {len(customers):,} rows | transactions.csv: {len(tx):,} rows")


def load_raw_to_postgres() -> None:
    """Load raw CSVs into PostgreSQL via SQLAlchemy (schema in sql/01_schema.sql)."""
    from sqlalchemy import create_engine, text

    engine = create_engine(config.DATABASE_URL)
    schema = (config.SQL_DIR / "01_schema.sql").read_text()
    with engine.begin() as conn:
        conn.execute(text(schema))
    customers = pd.read_csv(config.DATA_RAW / "customers.csv", parse_dates=["signup_date"])
    tx = pd.read_csv(config.DATA_RAW / "transactions.csv", parse_dates=["order_date"])
    customers.to_sql("customers", engine, if_exists="append", index=False)
    tx.to_sql("transactions_raw", engine, if_exists="append", index=False, chunksize=5000)
    print("Loaded customers and transactions_raw into PostgreSQL")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_raw_data()
    load_raw_to_postgres()
