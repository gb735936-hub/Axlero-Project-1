-- Flattened, user-level extract: one row per customer, using only
-- transactions strictly BEFORE the snapshot date (:snapshot) so the same
-- query can feed both EDA and leakage-free modeling.
-- Parameters: :snapshot (timestamp)
WITH base AS (
    SELECT t.*,
           t.unit_price * ABS(t.quantity) * (1 - t.discount_pct) AS line_value
    FROM v_transactions_clean t
    WHERE t.order_date < :snapshot
),
orders AS (
    SELECT customer_id, order_id, MIN(order_date) AS order_date,
           SUM(CASE WHEN quantity > 0 THEN line_value ELSE 0 END) AS order_value,
           COUNT(*)                                             AS n_lines,
           BOOL_OR(quantity < 0)                                AS has_return
    FROM base
    GROUP BY customer_id, order_id
)
SELECT c.customer_id,
       c.signup_date,
       c.city,
       c.acquisition_channel,
       COUNT(o.order_id)                                   AS n_orders,
       MIN(o.order_date)                                   AS first_order_date,
       MAX(o.order_date)                                   AS last_order_date,
       SUM(o.order_value)                                  AS total_spend,
       AVG(o.order_value)                                  AS avg_order_value,
       AVG(o.n_lines)                                      AS avg_lines_per_order,
       SUM(CASE WHEN o.has_return THEN 1 ELSE 0 END)       AS n_returns
FROM customers c
LEFT JOIN orders o USING (customer_id)
WHERE c.signup_date < :snapshot
GROUP BY c.customer_id, c.signup_date, c.city, c.acquisition_channel
ORDER BY c.customer_id;
