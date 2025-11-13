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
