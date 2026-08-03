# Databricks evidence checklist (Tier 2)

These three pieces of evidence require a live Databricks workspace with Unity
Catalog enabled, which isn't available in the environment these commits were
built and tested in. Everything below has been verified as correct in isolation
(dbt models parse cleanly against this exact schema, the notebooks are
syntax-checked, the Feature Store/MLflow API calls match the current SDK), but
none of it has been run end-to-end against a live workspace. Run these against
your own workspace and drop the screenshots into `docs/screenshots/` following
the existing `06_`, `07_`, `08_` numbering, then add them to the Evidence
section of the README.

This workspace uses `catalog=workspace`, `schema=fraud_feature_store` (see
[WORKSPACE_SETUP.md](../WORKSPACE_SETUP.md)). Items 6 and 8 already have real data
sitting in Unity Catalog from a previous session — you may not need to run
anything at all to capture those two; just open the tables and screenshot them.

## 6. Feature Store table in Unity Catalog

`workspace.fraud_feature_store.user_fraud_features` already has 1,000 rows from a
previous session — nothing needs to run first for this one.

1. In the workspace UI: **Catalog → `workspace` → `fraud_feature_store` → `user_fraud_features`**
2. Screenshot: the **Sample Data** tab (shows `user_id` as the primary key plus the
   velocity/spend/volatility aggregate columns) and the **Details** tab (shows row
   count, and that it's registered as a Feature Store table, not a plain Delta table).

## 7. MLflow experiment with metrics

This one does need `notebooks/02_model_training.py` to run first (see
WORKSPACE_SETUP.md for run order) — there's no existing v3 experiment yet.

1. After it runs, open **Experiments** in the workspace sidebar and find
   `/Shared/fraud-detection-feature-store-v3`.
2. Open the `gbm_time_split_smote` run.
3. Screenshot: the run's **Metrics** panel (`roc_auc`, `recall`, `precision`, `f1`,
   `avg_precision`) and, separately, the **Artifacts** tab showing
   `feature_importances.json` and the logged `fraud_model`.
4. Note in the README caption what ROC-AUC/recall the run actually produced —
   don't carry over the target numbers from the spec as if they were measured.

## 8. Bronze Delta table with streaming data

`workspace.fraud_feature_store.bronze_transactions` already has 284,807 rows from a
previous session (built by an earlier batch load, not the Kafka streaming job in
this repo — see WORKSPACE_SETUP.md on why notebook `00` is skipped). Nothing needs
to run first for this one either.

1. In the workspace UI: **Catalog → `workspace` → `fraud_feature_store` → `bronze_transactions`**
2. Screenshot: the **Details** tab (row count, Delta table, partitioning) and the
   **Sample Data** tab showing real ingested rows.
