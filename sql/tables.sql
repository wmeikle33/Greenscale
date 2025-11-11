SELECT COUNT(*) AS n,
       AVG(TRL_cm) AS mean,
       STDDEV_SAMP(TRL_cm) AS sd,
       MIN(TRL_cm) AS minv,
       MAX(TRL_cm) AS maxv
FROM t
WHERE value IS NOT NULL;
