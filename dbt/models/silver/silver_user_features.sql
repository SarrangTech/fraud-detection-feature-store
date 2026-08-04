{{ config(materialized='table', file_format='delta') }}

-- One row per user_id: the table registered into the Databricks Feature Store
-- (notebooks/01_feature_store_registration.py) and synced to Redis for online
-- serving (notebooks/03_redis_feature_sync.py). This is the primary-key = user_id
-- table the fraud-detection spec asks for.

select
    user_id,

    count(transaction_id) as total_txn_count,
    max(event_time) as last_transaction_time,

    avg(amount) as avg_amount_all_time,
    max(amount) as max_amount_all_time,
    stddev(amount) as amount_stddev_all_time,
    sum(amount) as total_spend_all_time,

    avg(txn_count_1h) as avg_velocity_1h,
    avg(txn_count_24h) as avg_velocity_24h,
    avg(txn_count_7d) as avg_velocity_7d,
    max(txn_count_1h) as max_velocity_1h,

    avg(total_amount_24h) as avg_daily_spend,
    -- NOTE: named avg_amount_volatility, not avg_amount_volatility_24h -- this must
    -- match notebooks/01_feature_store_registration.py's aggregation and
    -- serving/feature_mapping.py's expected column name exactly. See the schema
    -- test in dbt/tests/ and README Known Limitations before renaming.
    avg(amount_stddev_24h) as avg_amount_volatility,
    avg(amount_vs_avg_24h) as avg_amount_ratio,

    -- Per-user baselines of the anonymized PCA features -- cheap behavioral
    -- fingerprint used at scoring time as a fallback when a transaction's own
    -- V-features look far from the user's historical norm.
    avg(V1) as avg_v1, avg(V2) as avg_v2, avg(V3) as avg_v3,
    avg(V4) as avg_v4, avg(V5) as avg_v5

from {{ ref('silver_transaction_features') }}
group by user_id
