# Databricks notebook source
# TITLE: 03_feature_store
# PURPOSE: Create Databricks Feature Store table with user-level features
# KEY PATTERN: One row per user, updated incrementally as new transactions arrive
# PROJECT: Fraud Detection Feature Store

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Config
CATALOG = "workspace"
SCHEMA = "fraud_feature_store"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_features"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.user_fraud_features"

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

fe = FeatureEngineeringClient()

# COMMAND ----------

# Read silver features
df_silver = spark.read.format("delta").table(SILVER_TABLE)
print(f"Silver records: {df_silver.count()}")

# COMMAND ----------

# Build user-level feature table
# KEY DESIGN DECISION: Aggregate to one row per user
# At scoring time: transaction arrives → lookup user_id → get all features instantly
# This is what enables sub-100ms fraud scoring

df_user_features = (
    df_silver.groupBy("user_id").agg(
        # Transaction history
        F.count("transaction_id").alias("total_txn_count"),
        F.max("timestamp").alias("last_transaction_time"),

        # Spend patterns
        F.avg("amount").alias("avg_amount_all_time"),
        F.max("amount").alias("max_amount_all_time"),
        F.stddev("amount").alias("amount_stddev_all_time"),
        F.sum("amount").alias("total_spend_all_time"),

        # Velocity patterns
        F.avg("txn_count_1h").alias("avg_velocity_1h"),
        F.avg("txn_count_24h").alias("avg_velocity_24h"),
        F.avg("txn_count_7d").alias("avg_velocity_7d"),
        F.max("txn_count_1h").alias("max_velocity_1h"),

        # Recent spend patterns
        F.avg("total_amount_24h").alias("avg_daily_spend"),
        F.avg("amount_stddev_24h").alias("avg_amount_volatility"),
        F.avg("amount_vs_avg_24h").alias("avg_amount_ratio"),

        # PCA feature averages (baseline behavior per user)
        F.avg("V1").alias("avg_v1"),
        F.avg("V2").alias("avg_v2"),
        F.avg("V3").alias("avg_v3"),
        F.avg("V4").alias("avg_v4"),
        F.avg("V5").alias("avg_v5"),
    )
    .fillna(0)
)

print(f"Users in feature table: {df_user_features.count()}")
print(f"Feature columns: {len(df_user_features.columns)}")
display(df_user_features.limit(5))

# COMMAND ----------

# Create or update feature store table
try:
    fe.create_table(
        name=FEATURE_TABLE,
        primary_keys=["user_id"],
        df=df_user_features,
        description="""
        User-level fraud detection features.
        Updated incrementally as new transactions arrive.
        Primary key: user_id.
        Features: velocity (txn counts), spend patterns, amount volatility, PCA baselines.
        Used for sub-100ms real-time fraud scoring.
        """
    )
    print(f"Feature table CREATED: {FEATURE_TABLE}")
except Exception as e:
    if "already exists" in str(e).lower():
        fe.write_table(
            name=FEATURE_TABLE,
            df=df_user_features,
            mode="overwrite"
        )
        print(f"Feature table UPDATED: {FEATURE_TABLE}")
    else:
        raise e

# COMMAND ----------

# Verify feature table
df_verify = fe.read_table(name=FEATURE_TABLE)
print(f"Feature table records: {df_verify.count()}")
print(f"Columns: {df_verify.columns}")
display(df_verify.limit(10))

# COMMAND ----------

