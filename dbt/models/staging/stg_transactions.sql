-- Cleans and dedupes raw bronze events. Both the Kafka producer's `acks=all` retry
-- behavior and Structured Streaming's at-least-once delivery guarantee can produce
-- duplicate transaction_id rows on rare occasions; window functions downstream are
-- pass-sensitive to duplicates (velocity counts inflate), so dedupe here once, in a
-- single place, rather than in every downstream model.

with source as (
    select * from {{ source('bronze', 'bronze_transactions') }}
),

deduped as (
    select
        *,
        row_number() over (
            partition by transaction_id
            order by _ingested_at asc
        ) as _dedupe_rank
    from source
)

select
    transaction_id,
    user_id,
    event_time,
    cast(event_time as long) as event_time_unix,
    cast(event_time as date) as event_date,
    amount,
    is_fraud,
    V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
    V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
    V21, V22, V23, V24, V25, V26, V27, V28
from deduped
where _dedupe_rank = 1
