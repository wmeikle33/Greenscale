SELECT COUNT(*) AS n,
       AVG(TRL_cm) AS mean,
       STDDEV_SAMP(TRL_cm) AS sd,
       MIN(TRL_cm) AS minv,
       MAX(TRL_cm) AS maxv
FROM t
WHERE TRL_cm IS NOT NULL;

CREATE TABLE plants (
  id BIGSERIAL PRIMARY KEY,
  species TEXT NOT NULL,
  captured_at TIMESTAMPTZ DEFAULT now()
);

SELECT * FROM plants;  
SELECT * FROM plants LIMIT 20;
SELECT count(*) FROM plants;  

SELECT
  county,
  COUNT(*)            AS num_orders,
  COUNT(DISTINCT user_id) AS num_customers,
  SUM(amount)         AS total_revenue,
  AVG(amount)         AS avg_order_value,
  MIN(amount)         AS min_order_value,
  MAX(amount)         AS max_order_value
FROM sales
GROUP BY county;
