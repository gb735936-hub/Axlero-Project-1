-- Light, deterministic SQL-side cleaning: drop exact duplicate rows and
-- rows without a usable timestamp. Statistical imputation stays in Python
-- (src/features.py) so it can be fitted on training data only.
CREATE OR REPLACE VIEW v_transactions_clean AS
SELECT DISTINCT
    order_id, customer_id, order_date, category,
    unit_price, quantity, channel, discount_pct
FROM transactions_raw
WHERE order_date IS NOT NULL;
