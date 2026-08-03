-- Velocity features: how many transactions has this user made recently.
-- Frames use RANGE BETWEEN ... AND 1 PRECEDING on purpose: they look back over prior
-- transactions only and explicitly exclude the current row, so a transaction is scored
-- using nothing but history that existed before it arrived (no label/future leakage).

with txns as (
    select * from {{ ref('stg_transactions') }}
)

select
    transaction_id,
    user_id,
    event_time,

    count(transaction_id) over (
        partition by user_id
        order by event_time_unix
        range between {{ var('window_1h_seconds') }} preceding and 1 preceding
    ) as txn_count_1h,

    count(transaction_id) over (
        partition by user_id
        order by event_time_unix
        range between {{ var('window_24h_seconds') }} preceding and 1 preceding
    ) as txn_count_24h,

    count(transaction_id) over (
        partition by user_id
        order by event_time_unix
        range between {{ var('window_7d_seconds') }} preceding and 1 preceding
    ) as txn_count_7d,

    -- Seconds since this user's previous transaction (null = first transaction on record)
    event_time_unix - lag(event_time_unix) over (
        partition by user_id order by event_time_unix
    ) as seconds_since_last_txn

from txns
