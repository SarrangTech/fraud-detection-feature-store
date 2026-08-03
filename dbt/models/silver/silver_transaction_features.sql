{{
    config(
        materialized='table',
        file_format='delta',
        partition_by=['is_fraud']
    )
}}

-- One row per transaction: base columns + velocity + spend-pattern + volatility
-- features, ready for model training (notebooks/02_model_training.py) and for
-- rolling up into the user-level Feature Store table (silver_user_features.sql).
--
-- NOTE ON RECOMPUTE COST: rolling-window features fundamentally require scanning a
-- user's full transaction history every run to stay correct -- there's no partial/
-- incremental shortcut for a plain batch window function without moving this stage
-- to stateful streaming aggregation. We accept a full-table rebuild here and rely on
-- the Redis online store (serving/redis_client.py) so that cost is paid once per dbt
-- run, not once per scoring request.

with base as (
    select * from {{ ref('stg_transactions') }}
),

velocity as (
    select * from {{ ref('int_velocity_features') }}
),

spend as (
    select * from {{ ref('int_spend_pattern_features') }}
),

volatility as (
    select * from {{ ref('int_volatility_features') }}
)

select
    base.transaction_id,
    base.user_id,
    base.event_time,
    base.event_date,
    base.amount,
    base.is_fraud,

    coalesce(velocity.txn_count_1h, 0) as txn_count_1h,
    coalesce(velocity.txn_count_24h, 0) as txn_count_24h,
    coalesce(velocity.txn_count_7d, 0) as txn_count_7d,
    velocity.seconds_since_last_txn,

    coalesce(spend.total_amount_1h, 0) as total_amount_1h,
    coalesce(spend.total_amount_24h, 0) as total_amount_24h,
    coalesce(spend.avg_amount_24h, base.amount) as avg_amount_24h,
    coalesce(spend.max_amount_24h, base.amount) as max_amount_24h,
    coalesce(spend.min_amount_24h, base.amount) as min_amount_24h,
    coalesce(spend.amount_vs_avg_24h, 1.0) as amount_vs_avg_24h,
    coalesce(spend.amount_vs_max_7d, 1.0) as amount_vs_max_7d,

    coalesce(volatility.amount_stddev_24h, 0) as amount_stddev_24h,
    coalesce(volatility.amount_stddev_7d, 0) as amount_stddev_7d,
    coalesce(volatility.amount_zscore_7d, 0) as amount_zscore_7d,

    base.V1, base.V2, base.V3, base.V4, base.V5,
    base.V6, base.V7, base.V8, base.V9, base.V10,
    base.V11, base.V12, base.V13, base.V14, base.V15,
    base.V16, base.V17, base.V18, base.V19, base.V20,
    base.V21, base.V22, base.V23, base.V24, base.V25,
    base.V26, base.V27, base.V28

from base
left join velocity using (transaction_id)
left join spend using (transaction_id)
left join volatility using (transaction_id)
