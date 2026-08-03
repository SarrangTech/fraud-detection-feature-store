# Architecture

## Data flow

```
creditcard.csv
      │  streaming/producer.py (row-by-row replay, simulated real time)
      │  assigns synthetic user_id, JSON-encodes, keyed by user_id
      ▼
Kafka topic: fraud.transactions.raw
      │  notebooks/00_bronze_streaming_consumer.py
      │  Spark Structured Streaming, append-only
      ▼
Delta bronze table: <catalog>.<schema>.bronze_transactions
      │  dbt (dbt/models/staging, dbt/models/intermediate, dbt/models/silver)
      │  dedupe -> velocity / spend-pattern / volatility rolling-window features
      ▼
Delta silver tables:
  silver_transaction_features   (1 row / transaction -- used for TRAINING)
  silver_user_features          (1 row / user_id     -- used for SERVING)
      │                                       │
      │ notebooks/02_model_training.py        │ notebooks/01_feature_store_registration.py
      │ time-split 80/20 + SMOTE + GBM         │ registers into Databricks Feature Store
      │ MLflow tracking + UC model registry    ▼
      ▼                                Feature Store table (PK: user_id)
UC model registry:                             │ notebooks/03_redis_feature_sync.py
  <catalog>.<schema>.fraud_gbm_classifier       ▼
      │                                 Redis: user_features:<user_id> (hash, TTL)
      │                                        │
      └──────────────► serving/scoring_api.py ◄┘
                        transaction arrives -> Redis lookup -> model.predict_proba
                        -> APPROVE / DECLINE, target: <100ms end-to-end
```

## Design decisions worth knowing before reading the code

### 1. Synthetic `user_id`

The Kaggle dataset is anonymized at the transaction level: `Time`, `V1..V28`, `Amount`,
`Class` only -- no card/account/customer identifier. A feature store keyed on `user_id`
needs one, so `streaming/producer.py::simulate_user_id` deterministically hashes each
row's `(row_index, time_offset)` into a fixed pool of `SIMULATED_USER_POOL_SIZE`
(default 5,000) simulated accounts. It's seeded/deterministic so re-running the
producer against the same CSV reproduces the same user_id assignment. This is a
documented, standard workaround for this specific dataset, not a hidden shortcut --
see `data/README.md`.

### 2. Why model training does NOT join the `user_id` Feature Store table

`silver_user_features` (the Feature Store table) is an **all-time** aggregate,
recomputed on every dbt run. Joining it into training rows by `user_id` alone would
leak the future into the past -- e.g. `avg_amount_all_time` for a January transaction
would silently include that same user's March transactions.

`silver_transaction_features` avoids this by construction: every rolling feature uses
`RANGE BETWEEN <window> PRECEDING AND 1 PRECEDING`, i.e. only transactions that
occurred strictly before the current one, for that same user. That's what makes it
safe to train on directly. The Feature Store / Redis path is reserved for *serving*,
where the tradeoff below applies.

### 3. Train/serve feature approximation (documented, not a bug)

Recomputing exact point-in-time rolling windows (last 1h/24h/7d counts and sums) for
a single incoming transaction in under 10ms isn't feasible without re-scanning that
user's transaction history on every request. So at serving time,
`serving/feature_mapping.py::build_feature_vector` substitutes the closest available
signal: this user's all-time rolling aggregates from the Feature Store, refreshed into
Redis on every pipeline run (every 15 minutes by default, see
`databricks/resources/jobs.yml`). The mapping from proxy fields to training feature
slots is explicit and centralized in one function specifically so this gap stays
visible and auditable, rather than silently blurring "exact" and "approximate"
features together.

A brand-new `user_id` with no Redis entry yet (first transaction ever) falls back to
a cold-start default (zeros / the transaction's own amount as its own baseline) rather
than a fabricated history.

### 4. Why `silver_transaction_features` is a full `table` rebuild, not incremental

Rolling-window features require scanning a user's complete history every run --
there's no partial-recompute shortcut for a plain batch window function once new rows
land anywhere in that user's timeline. `insert_overwrite`-style incremental
materializations only reduce what gets *written*, not what has to be *scanned*, so for
this dataset's size (284,807 rows) a full rebuild is simpler and no slower in
practice. The natural v2 evolution, if input volume grows enough to matter, is moving
this stage to Spark Structured Streaming stateful aggregation (`groupByKey` +
`flatMapGroupsWithState` or windowed aggregation with watermarking) so each micro-batch
only touches recently active users.

### 5. Databricks Model Serving vs. the custom FastAPI + Redis service

The spec asks for both a Databricks Model Serving endpoint and a sub-100ms real-time
scoring pipeline. Both are provided, for different consumers:

- `databricks/resources/model_serving_endpoint.yml` stands up a serverless endpoint
  for the Unity Catalog-registered model, for Databricks-native consumers (SQL AI
  functions, batch jobs, other notebooks).
- `serving/scoring_api.py` is the consumer-facing, latency-critical path described in
  the spec. A REST round trip to a Databricks Serverless endpoint typically costs
  30-80ms on its own before any Redis lookup or business logic runs, which eats most
  of a 100ms budget. Loading the same registered model in-process, next to the Redis
  client, keeps the whole pipeline (lookup + score + decision) inside that budget.

## Latency budget (target)

| Stage                          | Budget   |
|---------------------------------|----------|
| Redis feature lookup (HGETALL)  | < 10ms   |
| Feature vector assembly (pure Python) | < 1ms |
| Model inference (GBM, ~30 features) | < 15ms |
| **Total end-to-end**            | **< 100ms** |

Actual numbers from your environment: `serving/scoring_api.py`'s `/score` response
includes a `latency` breakdown on every call, and the service logs a warning any time
`total_ms` exceeds `SCORING_LATENCY_BUDGET_MS`.
