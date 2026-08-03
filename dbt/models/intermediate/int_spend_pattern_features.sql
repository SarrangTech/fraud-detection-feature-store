-- Spend-pattern features: how much this user tends to spend, and how the current
-- transaction's amount compares to their recent baseline.

with txns as (
    select * from {{ ref('stg_transactions') }}
),

windowed as (
    select
        transaction_id,
        user_id,
        event_time,
        amount,

        sum(amount) over (
            partition by user_id
            order by event_time_unix
            range between {{ var('window_1h_seconds') }} preceding and 1 preceding
        ) as total_amount_1h,

        sum(amount) over (
            partition by user_id
            order by event_time_unix
            range between {{ var('window_24h_seconds') }} preceding and 1 preceding
        ) as total_amount_24h,

        avg(amount) over (
            partition by user_id
            order by event_time_unix
            range between {{ var('window_24h_seconds') }} preceding and 1 preceding
        ) as avg_amount_24h,

        max(amount) over (
            partition by user_id
            order by event_time_unix
            range between {{ var('window_24h_seconds') }} preceding and 1 preceding
        ) as max_amount_24h,

        min(amount) over (
            partition by user_id
            order by event_time_unix
            range between {{ var('window_24h_seconds') }} preceding and 1 preceding
        ) as min_amount_24h,

        max(amount) over (
            partition by user_id
            order by event_time_unix
            range between {{ var('window_7d_seconds') }} preceding and 1 preceding
        ) as max_amount_7d

    from txns
)

select
    transaction_id,
    user_id,
    event_time,
    total_amount_1h,
    total_amount_24h,
    avg_amount_24h,
    max_amount_24h,
    min_amount_24h,
    max_amount_7d,
    -- How this transaction compares to the user's recent baseline; ratios spike sharply
    -- for card-testing / account-takeover style fraud patterns.
    amount / (coalesce(avg_amount_24h, amount) + 0.01) as amount_vs_avg_24h,
    amount / (coalesce(max_amount_7d, amount) + 0.01) as amount_vs_max_7d
from windowed
