# Databricks evidence checklist (Tier 2) — DONE

All three items below have been captured against the real workspace
(`catalog=workspace`, `schema=fraud_feature_store` — see
[WORKSPACE_SETUP.md](../WORKSPACE_SETUP.md)) and are now in the README's
Evidence section (`docs/screenshots/06*`–`08*`). This file is kept for
reference on how they were captured, and as a template if you ever need to
re-capture them (e.g. after retraining with new data).

## 6. Feature Store table in Unity Catalog — captured

`workspace.fraud_feature_store.user_fraud_features`: 1,000 rows
(`sql.statistics.numRows: "1000"`), registered as a Feature Store table
(PK: `user_id`). Captured via **Catalog → `workspace` → `fraud_feature_store`
→ `user_fraud_features`** → Overview / Sample Data / Details tabs.

## 7. MLflow experiment with metrics — captured

`/Shared/fraud-detection-feature-store-v3`, run `gbm_time_split_smote`,
registered to `workspace.default.fraud_detection_feature_store` (v1).
Measured: ROC-AUC 0.9728, recall 0.7755, precision 0.5846, F1 0.6667,
avg precision 0.7561 — captured via **Experiments** → the run's Overview
(model registration) and metrics/params tables.

## 8. Bronze Delta table with streaming data — captured

`workspace.fraud_feature_store.bronze_transactions`: 284,807 rows
(`sql.statistics.historyStats.1.numRows: "284807"`), built by an earlier
batch load (not the Kafka streaming job in this repo — see
WORKSPACE_SETUP.md on why `00_bronze_streaming_consumer.py` is skipped
against this particular table). Captured via **Catalog → `workspace` →
`fraud_feature_store` → `bronze_transactions`** → Overview / Sample Data /
Details tabs.
