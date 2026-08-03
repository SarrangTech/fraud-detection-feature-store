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

## 6. Feature Store table in Unity Catalog

1. Deploy and run the pipeline against your workspace:
   ```bash
   cd databricks && databricks bundle deploy -t dev
   databricks bundle run -t dev feature_pipeline_job
   ```
   (this runs `dbt run` → `notebooks/01_feature_store_registration.py` → `02` → `03`)
2. In the workspace UI: **Catalog → `fraud_detection` → `feature_store` → `user_fraud_features`**
3. Screenshot: the **Sample Data** tab (shows `user_id` as the primary key plus the
   velocity/spend/volatility aggregate columns) and the **Details** tab (shows row
   count, and that it's registered as a Feature Store table, not a plain Delta table).

## 7. MLflow experiment with metrics

1. After `notebooks/02_model_training.py` runs, open **Experiments** in the workspace
   sidebar and find `/Shared/fraud-detection-feature-store`.
2. Open the `gbm_time_split_smote` run.
3. Screenshot: the run's **Metrics** panel (`roc_auc`, `recall`, `precision`, `f1`,
   `avg_precision`) and, separately, the **Artifacts** tab showing
   `feature_importances.json` and the logged `fraud_model`.
4. Note in the README caption what ROC-AUC/recall the run actually produced —
   don't carry over the target numbers from the spec as if they were measured.

## 8. Bronze Delta table with streaming data

1. With `streaming/producer.py` running and `notebooks/00_bronze_streaming_consumer.py`
   deployed as the `bronze_streaming_job` (continuous), let it ingest for a few minutes.
2. In the workspace UI: **Catalog → `fraud_detection` → `feature_store` → `bronze_transactions`**
3. Screenshot: the **Details** tab (row count, Delta table, partitioning) and the
   **Sample Data** tab showing real ingested rows with `_ingested_at`/`_source` populated.
