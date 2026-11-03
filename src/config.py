"""Central configuration for the churn & segmentation project."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_DIR = PROJECT_ROOT / "sql"

RANDOM_STATE = 42

# Synthetic data window
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
N_CUSTOMERS = 5000

# Churn definition (leakage-safe, time-based snapshot)
# Features use ONLY transactions before SNAPSHOT_DATE.
# Label = customer made NO purchase in the CHURN_WINDOW_DAYS after the snapshot.
SNAPSHOT_DATE = "2025-10-31"
CHURN_WINDOW_DAYS = 60

# Database connection (override with DATABASE_URL env var)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://nikki:nikki@localhost:5432/retail_churn",
)
