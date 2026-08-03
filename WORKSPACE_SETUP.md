# Workspace setup (this deployment)

This repo's code defaults to a specific Databricks workspace. If you're pointing
it at a different workspace, override `UC_CATALOG`/`UC_SCHEMA` (and the
`uc_catalog`/`uc_schema` Asset Bundle variables) instead of editing code.

## Catalog / schema

| | |
|---|---|
| Catalog | `workspace` |
| Schema | `fraud_feature_store` |

## Tables that already exist and are reused as-is

| Table | Rows | Built by | Notebook |
|---|---|---|---|
| `workspace.fraud_feature_store.bronze_transactions` | 284,807 | an earlier batch CSV load (not the Kafka streaming job in this repo) | n/a — see note below |
| `workspace.fraud_feature_store.silver_features` | 284,807 | per-transaction rolling-window features (velocity/spend/volatility) | consumed by `02` |
| `workspace.fraud_feature_store.user_fraud_features` | 1,000 | per-user aggregate, already registered as a Feature Store table (PK: `user_id`) | consumed/reused by `01` |
| `workspace.default.fraud_detection_v2` | — | an earlier registered model (run ID `dc738315509843f3982cc59a97cdef23`) | not touched by this repo — see below |

None of these are recreated or overwritten by the notebooks below; `01`
specifically checks whether `user_fraud_features` already exists and, if so,
just reads it instead of rebuilding it.

## What changed to make this work against your workspace

Beyond the catalog/schema rename itself, three real mismatches had to be fixed —
not just renamed — because your workspace's tables were built by an earlier,
simpler version of these notebooks (single `silver_features` table, no dbt),
while this repo's current architecture assumes a dbt-built `silver_transaction_features`
+ `silver_user_features` split (see `docs/architecture.md`). Doing a literal
find-and-replace on the old names would have produced code that errors or
silently corrupts data:

- **`01_feature_store_registration.py`** no longer assumes a pre-built
  `silver_user_features` dbt table. It now aggregates `silver_features` by
  `user_id` itself (same logic as whatever originally built `user_fraud_features`)
  — but only if `user_fraud_features` doesn't already exist. Registering
  `silver_features` directly as the user_id-keyed Feature Store table (a literal
  rename) would have failed: `silver_features` has 284,807 rows with many rows
  per `user_id`, which violates the Feature Store's primary-key uniqueness
  requirement.
- **`02_model_training.py`** now reads the time column as `timestamp` (not
  `event_time` — that's this dbt-based repo's column name, not what's in your
  actual table) and no longer references `amount_zscore_7d` as a training
  feature, since `silver_features` doesn't have that column (it's a dbt-only
  volatility feature added after your table was built). Everything else about
  the model — algorithm, time-based 80/20 split, SMOTE, MLflow tracking — is
  unchanged.
- **Model registration**: registers to `workspace.default.fraud_detection_feature_store`
  (catalog.**default**, not catalog.schema — matching where your existing
  `fraud_detection_v2` model already lives) so it doesn't collide with that
  earlier model. MLflow experiment: `/Shared/fraud-detection-feature-store-v3`.

## Notebooks you can skip

- **`00_bronze_streaming_consumer.py`** — requires a running Kafka broker
  (`docker-compose up -d` + `streaming/producer.py`), which isn't part of this
  workspace-setup task. It also writes a different bronze schema (columns like
  `event_time`, `_kafka_partition`) than your existing `bronze_transactions`
  table (which has `timestamp`, no Kafka metadata) — so even if you did have a
  broker, pointing it at the same table as-is would hit a Delta schema mismatch.
  Skip it; `bronze_transactions` already has your data.

## Run order

```
01_feature_store_registration.py   ->   02_model_training.py   ->   03_redis_feature_sync.py
```

- **01**: reuses `user_fraud_features` (already exists) — effectively a no-op
  read + verify unless that table is ever dropped, in which case it rebuilds it
  from `silver_features`.
- **02**: trains on `silver_features`, registers to
  `workspace.default.fraud_detection_feature_store`.
- **03**: syncs `user_fraud_features` to Redis (`REDIS_HOST`/`REDIS_PORT` env
  vars, default `localhost:6379`).

`dbt run` is **not** required for this run order — dbt is only needed if you
want to rebuild `silver_features`-equivalent tables from scratch via the
streaming path (`00` → dbt), which isn't in scope here.
