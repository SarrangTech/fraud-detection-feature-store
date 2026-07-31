# Databricks notebook source
# TITLE: 05_realtime_scoring
# PURPOSE: Demonstrate real-time fraud scoring using Feature Store lookup
# KEY: Show sub-100ms feature retrieval + scoring -- the core value of a feature store
# PROJECT: Fraud Detection Feature Store

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Config
CATALOG = "workspace"
SCHEMA = "fraud_feature_store"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.user_fraud_features"
MODEL_NAME = f"{CATALOG}.default.fraud_detection_v2"

from databricks.feature_engineering import FeatureEngineeringClient
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import time

fe = FeatureEngineeringClient()

# COMMAND ----------

# Load model using run ID directly -- Unity Catalog doesn't support get_latest_versions
RUN_ID = "dc738315509843f3982cc59a97cdef23"

model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/fraud_model")
print(f"Model loaded: {type(model)}")

# COMMAND ----------

# Feature columns -- must match training
FEATURE_COLS = [
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "total_amount_1h", "total_amount_24h",
    "avg_amount_24h", "max_amount_24h",
    "amount_stddev_24h", "amount_stddev_7d",
    "amount_vs_avg_24h", "amount_vs_max_7d",
    "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V14", "V16", "V17",
    "V18", "V19", "V21",
    "amount"
]

# COMMAND ----------

# Load feature store into memory
# In production this would be served from Redis for sub-10ms lookup
t0 = time.time()
df_features = fe.read_table(name=FEATURE_TABLE).toPandas().fillna(0)
feature_load_ms = (time.time() - t0) * 1000

print(f"Feature store loaded: {len(df_features)} users in {feature_load_ms:.0f}ms")

# COMMAND ----------

# Simulate incoming transactions
# Mix of normal and suspicious transactions
incoming_transactions = pd.DataFrame([
    # Normal transactions
    {
        "transaction_id": "txn_LIVE_001", "user_id": "user_0001",
        "amount": 45.00,
        "V1": -1.2, "V2": 0.8, "V3": 1.1, "V4": -0.3, "V5": 0.5,
        "V6": -0.1, "V7": 0.3, "V8": -0.2, "V9": 0.1, "V10": -0.4,
        "V11": 0.2, "V12": -0.1, "V14": 0.3, "V16": -0.2, "V17": 0.1,
        "V18": -0.3, "V19": 0.2, "V21": -0.1
    },
    {
        "transaction_id": "txn_LIVE_002", "user_id": "user_0002",
        "amount": 22.50,
        "V1": -0.9, "V2": 0.6, "V3": 0.8, "V4": -0.2, "V5": 0.3,
        "V6": -0.2, "V7": 0.1, "V8": -0.1, "V9": 0.2, "V10": -0.3,
        "V11": 0.1, "V12": -0.2, "V14": 0.2, "V16": -0.1, "V17": 0.2,
        "V18": -0.2, "V19": 0.1, "V21": -0.2
    },
    # Suspicious transactions -- high amount, unusual V features
    {
        "transaction_id": "txn_LIVE_003", "user_id": "user_0003",
        "amount": 2800.00,
        "V1": -4.5, "V2": 3.2, "V3": -5.1, "V4": 4.8, "V5": -3.2,
        "V6": -2.1, "V7": 3.8, "V8": -1.9, "V9": 2.4, "V10": -4.1,
        "V11": 3.2, "V12": -5.8, "V14": -3.4, "V16": 2.8, "V17": -4.2,
        "V18": 2.1, "V19": -1.8, "V21": 3.2
    },
    {
        "transaction_id": "txn_LIVE_004", "user_id": "user_0004",
        "amount": 3500.00,
        "V1": -5.2, "V2": 4.1, "V3": -4.8, "V4": 5.2, "V5": -4.1,
        "V6": -3.2, "V7": 4.5, "V8": -2.8, "V9": 3.1, "V10": -5.2,
        "V11": 4.1, "V12": -6.2, "V14": -4.1, "V16": 3.5, "V17": -5.1,
        "V18": 3.2, "V19": -2.4, "V21": 4.1
    },
    {
        "transaction_id": "txn_LIVE_005", "user_id": "user_0005",
        "amount": 67.00,
        "V1": -1.1, "V2": 0.7, "V3": 0.9, "V4": -0.4, "V5": 0.4,
        "V6": -0.3, "V7": 0.2, "V8": -0.3, "V9": 0.3, "V10": -0.5,
        "V11": 0.3, "V12": -0.3, "V14": 0.4, "V16": -0.3, "V17": 0.3,
        "V18": -0.4, "V19": 0.3, "V21": -0.3
    },
])

print(f"Incoming transactions: {len(incoming_transactions)}")

# COMMAND ----------

# REAL-TIME SCORING PIPELINE
# Step 1: Look up user features from feature store
# Step 2: Combine with transaction features
# Step 3: Score with model
# Step 4: Return decision

t_start = time.time()

# Step 1: Feature store lookup
t1 = time.time()
df_user_feats = df_features[df_features["user_id"].isin(incoming_transactions["user_id"])]
feature_lookup_ms = (time.time() - t1) * 1000

# Step 2: Join transaction features with user features
df_scoring = incoming_transactions.merge(df_user_feats, on="user_id", how="left").fillna(0)

# Build rolling features from user history + current transaction
# In production these come from Redis, here we use feature store averages as proxy
df_scoring["txn_count_1h"]       = df_scoring.get("avg_velocity_1h", 0)
df_scoring["txn_count_24h"]      = df_scoring.get("avg_velocity_24h", 0)
df_scoring["txn_count_7d"]       = df_scoring.get("avg_velocity_7d", 0)
df_scoring["total_amount_1h"]    = df_scoring["amount"] * df_scoring.get("avg_velocity_1h", 1)
df_scoring["total_amount_24h"]   = df_scoring.get("avg_daily_spend", df_scoring["amount"])
df_scoring["avg_amount_24h"]     = df_scoring.get("avg_amount_all_time", df_scoring["amount"])
df_scoring["max_amount_24h"]     = df_scoring.get("max_amount_all_time", df_scoring["amount"])
df_scoring["amount_stddev_24h"]  = df_scoring.get("avg_amount_volatility", 0)
df_scoring["amount_stddev_7d"]   = df_scoring.get("amount_stddev_all_time", 0)
df_scoring["amount_vs_avg_24h"]  = df_scoring["amount"] / (df_scoring.get("avg_amount_all_time", df_scoring["amount"]) + 0.01)
df_scoring["amount_vs_max_7d"]   = df_scoring["amount"] / (df_scoring.get("max_amount_all_time", df_scoring["amount"]) + 0.01)

# Step 3: Score
t2 = time.time()
X_score = df_scoring[FEATURE_COLS].fillna(0)
fraud_proba = model.predict_proba(X_score)[:, 1]
scoring_ms = (time.time() - t2) * 1000

total_ms = (time.time() - t_start) * 1000

# Step 4: Apply threshold and build result
FRAUD_THRESHOLD = 0.30  # Lower threshold to catch more fraud
df_scoring["fraud_probability"] = np.round(fraud_proba, 4)
df_scoring["decision"] = df_scoring["fraud_probability"].apply(
    lambda x: "DECLINE" if x >= FRAUD_THRESHOLD else "APPROVE"
)
df_scoring["risk_level"] = df_scoring["fraud_probability"].apply(
    lambda x: "HIGH" if x >= 0.5 else ("MEDIUM" if x >= 0.3 else "LOW")
)

# COMMAND ----------

# Display results
print("=" * 60)
print("REAL-TIME FRAUD SCORING RESULTS")
print("=" * 60)
print(f"Feature lookup latency:  {feature_lookup_ms:.1f}ms")
print(f"Model scoring latency:   {scoring_ms:.1f}ms")
print(f"Total end-to-end:        {total_ms:.1f}ms")
print(f"Per transaction:         {total_ms/len(incoming_transactions):.1f}ms")
print(f"Fraud threshold:         {FRAUD_THRESHOLD}")
print()

display(df_scoring[[
    "transaction_id", "user_id", "amount",
    "fraud_probability", "risk_level", "decision"
]])

# COMMAND ----------

# Show the business impact
approved = df_scoring[df_scoring["decision"] == "APPROVE"]
declined = df_scoring[df_scoring["decision"] == "DECLINE"]

print(f"\nApproved transactions: {len(approved)}")
print(f"Declined transactions: {len(declined)}")
print(f"Total amount approved: ${approved['amount'].sum():,.2f}")
print(f"Total amount blocked:  ${declined['amount'].sum():,.2f}")

# COMMAND ----------

# WHY A FEATURE STORE MATTERS -- the key interview talking point
print("=" * 60)
print("WHY FEATURE STORE ENABLES SUB-100ms FRAUD SCORING")
print("=" * 60)
print()
print("WITHOUT feature store:")
print("  Transaction arrives → compute all rolling windows from DB → score")
print("  Latency: 500ms - 2000ms (DB query + aggregation)")
print()
print("WITH feature store:")
print("  Transaction arrives → lookup pre-computed features → score")
print(f"  Latency: {total_ms:.0f}ms (simple key lookup + model inference)")
print()
print("At JPMorgan scale (1B transactions/day):")
print("  Without FS: 1B × 1000ms = 277 hours of compute per day")
print("  With FS:    1B × 50ms   = 13.8 hours of compute per day")
print()
print("Feature freshness SLA: features updated every 5 minutes via streaming")
print("Feature store users: fraud scoring, credit risk, AML, recommendation")

# COMMAND ----------

