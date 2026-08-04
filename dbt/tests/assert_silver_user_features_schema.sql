-- Singular test: fails to compile (and therefore fails `dbt test`) if any of these
-- columns is missing or renamed in silver_user_features. These are exactly the
-- column names serving/feature_mapping.py and notebooks/01_feature_store_registration.py
-- expect -- see README Known Limitations on why this model and that notebook must be
-- kept in sync manually. `where 1 = 0` means this returns zero rows (passes) as long
-- as every referenced column resolves.
select
    user_id,
    total_txn_count,
    last_transaction_time,
    avg_amount_all_time,
    max_amount_all_time,
    amount_stddev_all_time,
    total_spend_all_time,
    avg_velocity_1h,
    avg_velocity_24h,
    avg_velocity_7d,
    max_velocity_1h,
    avg_daily_spend,
    avg_amount_volatility,
    avg_amount_ratio,
    avg_v1, avg_v2, avg_v3, avg_v4, avg_v5
from {{ ref('silver_user_features') }}
where 1 = 0
