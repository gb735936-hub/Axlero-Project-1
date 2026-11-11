-- Raw layer: loaded as-is from CSV (no cleaning).
DROP VIEW  IF EXISTS v_transactions_clean CASCADE;
DROP TABLE IF EXISTS transactions_raw CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

CREATE TABLE customers (
    customer_id          INTEGER PRIMARY KEY,
    signup_date          TIMESTAMP NOT NULL,
    city                 VARCHAR(50),
    acquisition_channel  VARCHAR(30)
);

CREATE TABLE transactions_raw (
    order_id      INTEGER,
    customer_id   INTEGER REFERENCES customers (customer_id),
    order_date    TIMESTAMP,
    category      VARCHAR(50),
    unit_price    NUMERIC(12, 2),
    quantity      NUMERIC(6, 0),
    channel       VARCHAR(30),
    discount_pct  NUMERIC(4, 2)
);

CREATE INDEX idx_tx_customer ON transactions_raw (customer_id);
CREATE INDEX idx_tx_date     ON transactions_raw (order_date)
