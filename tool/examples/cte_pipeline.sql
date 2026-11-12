WITH FIRST_TXN AS (
    SELECT customer_id, MIN(txn_date) AS first_txn_date
    FROM customer_transactions
    GROUP BY customer_id
),
FILTERED_30D AS (
    SELECT c.customer_id, c.txn_date,
           CASE WHEN txn_type = 'deposit' THEN txn_amount ELSE -txn_amount END AS amount
    FROM customer_transactions c
    JOIN FIRST_TXN f ON c.customer_id = f.customer_id
    WHERE c.txn_date >= DATE(f.first_txn_date, '+30 days')
),
BALANCES AS (
    SELECT customer_id, txn_date,
           SUM(amount) OVER (PARTITION BY customer_id ORDER BY txn_date ROWS UNBOUNDED PRECEDING) AS bal
    FROM FILTERED_30D
),
ROLLING_AVG AS (
    SELECT customer_id, txn_date,
           AVG(bal) OVER (PARTITION BY customer_id ORDER BY txn_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS avg_bal
    FROM BALANCES
),
ZEROED_AVG AS (
    SELECT *, CASE WHEN avg_bal < 0 THEN 0 ELSE avg_bal END AS zeroed_avg
    FROM ROLLING_AVG
),
WITH_MONTH AS (
    SELECT *, CAST(strftime('%Y', txn_date) AS INTEGER) AS cal_year,
              CAST(strftime('%m', txn_date) AS INTEGER) AS cal_month
    FROM ZEROED_AVG
)
SELECT customer_id, cal_year, cal_month, MAX(zeroed_avg) AS mx_avg
FROM WITH_MONTH
GROUP BY customer_id, cal_year, cal_month;
