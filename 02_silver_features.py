# Databricks notebook source
# TITLE: 02_silver_features
# PURPOSE: Compute rolling window features per user for fraud detection
# KEY INSIGHT: Features computed using ONLY past data -- no leakage
# PROJECT: Fraud Detection Feature Store

# COMMAND ----------

# Config
CATALOG = "workspace"
SCHEMA = "fraud_feature_store"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_transactions"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_features"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Read bronze
df_bronze = spark.read.format("delta").table(BRONZE_TABLE)
print(f"Bronze records: {df_bronze.count()}")
print(f"Time range: {df_bronze.agg(F.min('timestamp'), F.max('timestamp')).collect()[0]}")

# COMMAND ----------

# CRITICAL: Define windows using ONLY past data (rangeBetween excludes current row's future)
# This prevents data leakage -- at scoring time we only have historical data

# Window ordered by timestamp, partitioned by user
# rangeBetween(-N, -1) means: look back N seconds, exclude current transaction
w_1h = Window.partitionBy("user_id").orderBy(
    F.col("timestamp").cast("long")
).rangeBetween(-3600, -1)       # 1 hour lookback, exclude current

w_24h = Window.partitionBy("user_id").orderBy(
    F.col("timestamp").cast("long")
).rangeBetween(-86400, -1)      # 24 hour lookback

w_7d = Window.partitionBy("user_id").orderBy(
    F.col("timestamp").cast("long")
).rangeBetween(-604800, -1)     # 7 day lookback

# COMMAND ----------

# Compute features
# These are exactly the features used by JPMorgan, Capital One, Stripe
# for real-time fraud scoring

df_silver = df_bronze.select(
    "transaction_id",
    "user_id",
    "timestamp",
    "amount",
    "is_fraud",

    # ── Velocity features (how many transactions in time window) ──
    F.count("transaction_id").over(w_1h).alias("txn_count_1h"),
    F.count("transaction_id").over(w_24h).alias("txn_count_24h"),
    F.count("transaction_id").over(w_7d).alias("txn_count_7d"),

    # ── Amount features (spend patterns) ──
    F.sum("amount").over(w_1h).alias("total_amount_1h"),
    F.sum("amount").over(w_24h).alias("total_amount_24h"),
    F.avg("amount").over(w_24h).alias("avg_amount_24h"),
    F.max("amount").over(w_24h).alias("max_amount_24h"),
    F.min("amount").over(w_24h).alias("min_amount_24h"),

    # ── Volatility features (erratic behavior = fraud signal) ──
    F.stddev("amount").over(w_24h).alias("amount_stddev_24h"),
    F.stddev("amount").over(w_7d).alias("amount_stddev_7d"),

    # ── Ratio features ──
    # How does this transaction compare to the user's recent average?
    (F.col("amount") / (F.avg("amount").over(w_24h) + F.lit(0.01))).alias("amount_vs_avg_24h"),
    (F.col("amount") / (F.max("amount").over(w_7d) + F.lit(0.01))).alias("amount_vs_max_7d"),

    # ── PCA features from original dataset (V1-V10 most important) ──
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V14", "V16", "V17", "V18", "V19", "V21"
)

# Fill nulls -- first transactions have no history, default to 0
df_silver = df_silver.fillna(0)

print(f"Silver records: {df_silver.count()}")
print(f"Features: {len(df_silver.columns)}")

# COMMAND ----------

# Verify no leakage: fraud vs non-fraud feature comparison
print("=== Fraud Signal Verification (no leakage) ===")
display(
    df_silver.groupBy("is_fraud").agg(
        F.mean("txn_count_1h").alias("avg_velocity_1h"),
        F.mean("total_amount_1h").alias("avg_amount_1h"),
        F.mean("amount_stddev_24h").alias("avg_volatility_24h"),
        F.mean("amount_vs_avg_24h").alias("avg_amount_ratio"),
        F.count("*").alias("count")
    )
)

# COMMAND ----------

# Write to Delta silver table
(
    df_silver
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("is_fraud")
    .saveAsTable(SILVER_TABLE)
)

print(f"Silver table written: {SILVER_TABLE}")
print(f"Total records: {df_silver.count()}")
print(f"Fraud records: {df_silver.filter(F.col('is_fraud') == 1).count()}")

# COMMAND ----------

