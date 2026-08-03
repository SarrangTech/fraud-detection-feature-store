# Databricks notebook source
# TITLE: 02_model_training
# PURPOSE: Train a GradientBoostingClassifier on dbt's silver_transaction_features table
#          using a time-based 80/20 split and SMOTE for class imbalance, tracked in MLflow
#          and registered to the Unity Catalog model registry.
#
# WHY TRAINING DOES NOT JOIN THE user_id FEATURE STORE TABLE:
#   silver_user_features (the Feature Store table, PK=user_id) is an ALL-TIME aggregate
#   recomputed on every dbt run. Joining it into training rows by user_id alone would leak
#   the future into the past (e.g. avg_amount_all_time for a January transaction would
#   include that user's March transactions). silver_transaction_features avoids this by
#   construction -- every rolling feature uses RANGE BETWEEN ... PRECEDING, i.e. only data
#   that existed strictly before that transaction. So training uses the exact,
#   leakage-safe per-transaction features; the user_id Feature Store table is reserved for
#   the ONLINE serving path (synced to Redis), where an always-fresh aggregate is the best
#   available proxy at sub-10ms latency. See docs/architecture.md for the full rationale.
# PROJECT: Fraud Detection Feature Store

# COMMAND ----------

# MAGIC %pip install imbalanced-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("uc_catalog", "fraud_detection")
dbutils.widgets.text("uc_schema", "feature_store")
dbutils.widgets.text("mlflow_experiment", "/Shared/fraud-detection-feature-store")

CATALOG = dbutils.widgets.get("uc_catalog")
SCHEMA = dbutils.widgets.get("uc_schema")
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_transaction_features"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_gbm_classifier"

from pyspark.sql import functions as F
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(dbutils.widgets.get("mlflow_experiment"))

# COMMAND ----------

# ── TIME-BASED 80/20 SPLIT ───────────────────────────────────────────────────
# Train on the first 80% of the time range, test on the last 20%: the model is
# always evaluated on transactions strictly *after* everything it trained on,
# which is the only split that resembles how the model will actually be used.

df_silver = spark.read.table(SILVER_TABLE)

time_stats = df_silver.agg(
    F.min("event_time").alias("min_time"), F.max("event_time").alias("max_time")
).collect()[0]
min_time, max_time = time_stats["min_time"], time_stats["max_time"]
split_time = min_time + (max_time - min_time) * 0.80

print(f"Dataset time range: {min_time} -> {max_time}")
print(f"Train: {min_time} -> {split_time}")
print(f"Test:  {split_time} -> {max_time}")

df_train_base = df_silver.filter(F.col("event_time") < F.lit(split_time))
df_test_base = df_silver.filter(F.col("event_time") >= F.lit(split_time))

print(f"Train records: {df_train_base.count()}, fraud: {df_train_base.filter('is_fraud=1').count()}")
print(f"Test records:  {df_test_base.count()}, fraud: {df_test_base.filter('is_fraud=1').count()}")

# COMMAND ----------

FEATURE_COLS = [
    # Velocity
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    # Spend patterns
    "total_amount_1h", "total_amount_24h", "avg_amount_24h", "max_amount_24h", "min_amount_24h",
    "amount_vs_avg_24h", "amount_vs_max_7d",
    # Volatility
    "amount_stddev_24h", "amount_stddev_7d", "amount_zscore_7d",
    # Most predictive PCA components for this dataset + the raw amount
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V14", "V16", "V17", "V18", "V19", "V21",
    "amount",
]

df_train_pd = df_train_base.select(FEATURE_COLS + ["is_fraud"]).toPandas().fillna(0)
df_test_pd = df_test_base.select(FEATURE_COLS + ["is_fraud"]).toPandas().fillna(0)

X_train, y_train = df_train_pd[FEATURE_COLS], df_train_pd["is_fraud"]
X_test, y_test = df_test_pd[FEATURE_COLS], df_test_pd["is_fraud"]

print(f"X_train: {X_train.shape}, fraud rate: {y_train.mean():.4%}")
print(f"X_test:  {X_test.shape}, fraud rate: {y_test.mean():.4%}")

# COMMAND ----------

# ── SMOTE (train split only -- never resample the held-out test set) ────────
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_bal.shape}, fraud rate: {y_train_bal.mean():.4%}")

# COMMAND ----------

FRAUD_THRESHOLD = 0.30  # lower than 0.5 on purpose: recall matters more than precision here

with mlflow.start_run(run_name="gbm_time_split_smote") as run:
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train_bal, y_train_bal)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= FRAUD_THRESHOLD).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_fraud_count": int(y_test.sum()),
    }

    mlflow.log_params({
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
        "min_samples_leaf": 20, "smote_sampling_strategy": 0.1,
        "split_method": "time_based_80_20", "fraud_threshold": FRAUD_THRESHOLD,
        "n_features": len(FEATURE_COLS),
    })
    mlflow.log_metrics(metrics)

    importances = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    mlflow.log_dict(importances.to_dict("records"), "feature_importances.json")

    signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="fraud_model",
        registered_model_name=MODEL_NAME,
        input_example=X_train.head(5),
        signature=signature,
    )

    print("=" * 50)
    print(f"ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"Avg Precision:  {metrics['avg_precision']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1:             {metrics['f1']:.4f}")
    print(f"Run ID:         {run.info.run_id}")
    print(f"Registered as:  {MODEL_NAME}")

# COMMAND ----------

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix (rows=actual, cols=predicted [not-fraud, fraud]):")
print(cm)
display(importances.head(10))
