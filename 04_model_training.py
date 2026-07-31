# Databricks notebook source
# TITLE: 04_model_training
# PURPOSE: Train fraud detection model using Feature Store with time-based train/test split
# KEY: Time-based split prevents data leakage -- model never sees future data during training
# PROJECT: Fraud Detection Feature Store

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering imbalanced-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Config
CATALOG = "workspace"
SCHEMA = "fraud_feature_store"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_features"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.user_fraud_features"
MODEL_NAME = f"{CATALOG}.default.fraud_detection_v2"

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql import functions as F
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

fe = FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# TIME-BASED TRAIN/TEST SPLIT
# This is the correct way to prevent data leakage in time-series fraud data
# Train on first 80% of time, test on last 20%
# This simulates production: model trained on past, scored on future

df_silver = spark.read.format("delta").table(SILVER_TABLE)

# Get the time split point (80th percentile of timestamps)
time_stats = df_silver.agg(
    F.min("timestamp").alias("min_time"),
    F.max("timestamp").alias("max_time")
).collect()[0]

min_time = time_stats["min_time"]
max_time = time_stats["max_time"]
time_range_seconds = (max_time - min_time).total_seconds()
split_time = min_time + pd.Timedelta(seconds=time_range_seconds * 0.80)

print(f"Dataset time range: {min_time} → {max_time}")
print(f"Train/test split at: {split_time}")
print(f"Train: {min_time} → {split_time}")
print(f"Test:  {split_time} → {max_time}")

# COMMAND ----------

# Create train and test base DataFrames
df_train_base = df_silver.filter(F.col("timestamp") < F.lit(split_time))
df_test_base  = df_silver.filter(F.col("timestamp") >= F.lit(split_time))

print(f"Train records: {df_train_base.count()}")
print(f"Test records:  {df_test_base.count()}")
print(f"Train fraud:   {df_train_base.filter(F.col('is_fraud')==1).count()}")
print(f"Test fraud:    {df_test_base.filter(F.col('is_fraud')==1).count()}")

# COMMAND ----------

# Feature columns to use for training
# Mix of rolling window features + PCA features from dataset
FEATURE_COLS = [
    # Rolling window features (computed in silver)
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "total_amount_1h", "total_amount_24h",
    "avg_amount_24h", "max_amount_24h",
    "amount_stddev_24h", "amount_stddev_7d",
    "amount_vs_avg_24h", "amount_vs_max_7d",

    # PCA features (most predictive from original dataset)
    "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V14", "V16", "V17",
    "V18", "V19", "V21",

    # Amount
    "amount"
]

# Convert to pandas for sklearn
df_train_pd = df_train_base.select(FEATURE_COLS + ["is_fraud"]).toPandas().fillna(0)
df_test_pd  = df_test_base.select(FEATURE_COLS + ["is_fraud"]).toPandas().fillna(0)

X_train = df_train_pd[FEATURE_COLS]
y_train = df_train_pd["is_fraud"]
X_test  = df_test_pd[FEATURE_COLS]
y_test  = df_test_pd["is_fraud"]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"Train fraud rate: {y_train.mean():.4%}")
print(f"Test fraud rate:  {y_test.mean():.4%}")

# COMMAND ----------

# Handle class imbalance with SMOTE
# Fraud is ~0.17% of transactions -- without SMOTE model predicts everything as non-fraud
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"After SMOTE -- Train shape: {X_train_balanced.shape}")
print(f"After SMOTE -- Fraud rate: {y_train_balanced.mean():.4%}")

# COMMAND ----------

# Train model with MLflow tracking
mlflow.set_experiment("/Shared/fraud-detection-feature-store-v2")

with mlflow.start_run(run_name="gradient_boosting_time_split_v1") as run:

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42
    )

    model.fit(X_train_balanced, y_train_balanced)

    # Evaluate on held-out TEST set (future data model never saw)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_50    = (y_pred_proba >= 0.50).astype(int)
    y_pred_30    = (y_pred_proba >= 0.30).astype(int)  # lower threshold for fraud detection

    roc_auc  = roc_auc_score(y_test, y_pred_proba)
    avg_prec = average_precision_score(y_test, y_pred_proba)  # better metric for imbalanced
    prec_50  = precision_score(y_test, y_pred_50, zero_division=0)
    rec_50   = recall_score(y_test, y_pred_50, zero_division=0)
    f1_50    = f1_score(y_test, y_pred_50, zero_division=0)
    prec_30  = precision_score(y_test, y_pred_30, zero_division=0)
    rec_30   = recall_score(y_test, y_pred_30, zero_division=0)

    # Log params
    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "smote_strategy": 0.1,
        "split_method": "time_based_80_20",
        "n_features": len(FEATURE_COLS)
    })

    # Log metrics
    mlflow.log_metrics({
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
        "precision_t50": prec_50,
        "recall_t50": rec_50,
        "f1_t50": f1_50,
        "precision_t30": prec_30,
        "recall_t30": rec_30,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_fraud_count": int(y_test.sum())
    })

    # Log feature importances
    importances = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    mlflow.log_dict(
        importances.head(10).to_dict("records"),
        "top_10_features.json"
    )

    # Register model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="fraud_model",
        registered_model_name=MODEL_NAME,
        input_example=X_train.head(5)
    )

    print("=" * 50)
    print("MODEL EVALUATION (time-based test split)")
    print("=" * 50)
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"Avg Precision (AUPRC):{avg_prec:.4f}")
    print(f"\nAt threshold 0.50:")
    print(f"  Precision:          {prec_50:.4f}")
    print(f"  Recall:             {rec_50:.4f}")
    print(f"  F1:                 {f1_50:.4f}")
    print(f"\nAt threshold 0.30 (lower = catch more fraud):")
    print(f"  Precision:          {prec_30:.4f}")
    print(f"  Recall:             {rec_30:.4f}")
    print(f"\nRun ID: {run.info.run_id}")

# COMMAND ----------

# Show top feature importances
print("\nTop 10 Most Important Features:")
display(importances.head(10))

# COMMAND ----------

# Confusion matrix at threshold 0.30
cm = confusion_matrix(y_test, y_pred_30)
print("\nConfusion Matrix (threshold=0.30):")
print(f"                 Predicted NOT Fraud  Predicted FRAUD")
print(f"Actual NOT Fraud      {cm[0][0]:>10}          {cm[0][1]:>10}")
print(f"Actual FRAUD          {cm[1][0]:>10}          {cm[1][1]:>10}")
print(f"\nTrue Positives (fraud caught):    {cm[1][1]}")
print(f"False Negatives (fraud missed):   {cm[1][0]}")
print(f"False Positives (false alarms):   {cm[0][1]}")

# COMMAND ----------

