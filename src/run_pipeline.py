"""One-command reproduction of every data/model artifact:

python -m src.dataset          # raw CSVs + load to PostgreSQL
python -m src.run_pipeline     # interim -> processed -> models -> BI exports
"""

from __future__ import annotations

import json
import warnings

import joblib
import pandas as pd

from src import config, features as F
from src.modeling import predict as P, segment as S, train as T

warnings.filterwarnings("ignore")


def main() -> None:
    for d in (config.DATA_INTERIM, config.DATA_PROCESSED, config.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Week 1: SQL extraction + cleaning -> data/interim
    raw = F.load_clean_view()
    tx, audit = F.clean_transactions(raw)
    tx.to_csv(config.DATA_INTERIM / "transactions_clean.csv", index=False)
    F.extract_user_level().to_csv(config.DATA_INTERIM / "user_level_extract.csv", index=False)
    (config.DATA_INTERIM / "cleaning_audit.json").write_text(json.dumps(audit, indent=2))
    print("cleaning audit:", audit)

    customers = pd.read_csv(config.DATA_RAW / "customers.csv", parse_dates=["signup_date"])

    # ---- Week 2: RFM + K-Means -> data/processed
    table = F.assemble_model_table(tx, customers, config.SNAPSHOT_DATE)
    k_scores = S.evaluate_k(table)
    best_k = int(k_scores[k_scores["k"] >= 4].sort_values("silhouette").iloc[-1]["k"])
    seg_pipe = S.make_pipeline(best_k).fit(table[F.RFM_COLUMNS])
    names = S.name_clusters(table, seg_pipe.predict(table[F.RFM_COLUMNS]))
    table["segment"] = [names[c] for c in seg_pipe.predict(table[F.RFM_COLUMNS])]
    table.to_csv(config.DATA_PROCESSED / "customer_segments.csv", index=False)
    S.segment_summary(table).to_csv(config.DATA_PROCESSED / "segment_summary.csv")
    joblib.dump(seg_pipe, config.MODELS_DIR / "kmeans_rfm_pipeline.joblib")
    joblib.dump(names, config.MODELS_DIR / "segment_names.joblib")
    print(f"k={best_k} segments:", sorted(names.values()))

    # ---- Week 3: churn models -> models/
    results, fitted, X_te, y_te = T.run_experiment(table)
    results.to_csv(config.DATA_PROCESSED / "model_comparison.csv", index=False)
    best = results.iloc[0]["model"]
    T.save_model(fitted[best], "churn_model", T.evaluate(fitted[best], X_te, y_te))
    print(results.round(3).to_string())

    # ---- Week 4: score live base -> BI exports
    scoring_date = (pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    seg_m, names_m, churn_m = P.load_models()
    scores = P.score_customers(tx, customers, scoring_date, seg_m, names_m, churn_m)
    scores.to_csv(config.DATA_PROCESSED / "customer_scores_for_bi.csv", index=False)
    valid = tx[tx["order_date"] < scoring_date].copy()
    valid["month"] = valid["order_date"].dt.to_period("M").dt.to_timestamp()
    seg_lookup = scores.set_index("customer_id")["segment"]
    valid["segment"] = valid["customer_id"].map(seg_lookup)
    (
        valid.groupby(["month", "segment"], as_index=False)
        .agg(revenue=("line_value", "sum"), orders=("order_id", "nunique"))
        .to_csv(config.DATA_PROCESSED / "monthly_revenue_by_segment.csv", index=False)
    )
    print("high-risk customers:", int((scores["risk_tier"] == "High").sum()))


if __name__ == "__main__":
    main()
