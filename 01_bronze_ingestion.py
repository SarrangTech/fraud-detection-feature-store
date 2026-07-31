# Databricks notebook source
# TITLE: 01_bronze_ingestion
# PURPOSE: Ingest raw Kaggle credit card fraud dataset into Delta bronze table
# PROJECT: Fraud Detection Feature Store

# COMMAND ----------

# Config
CATALOG = "workspace"
SCHEMA = "fraud_feature_store"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_transactions"
RAW_PATH = "/Volumes/workspace/fraud_feature_store/raw_data/creditcard.csv"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

# Read raw CSV
df_raw = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(RAW_PATH)
)

print(f"Raw records: {df_raw.count()}")
print(f"Columns: {df_raw.columns}")

# COMMAND ----------

# Add ingestion metadata and rename columns for clarity
# The dataset has columns: Time, V1-V28 (PCA features), Amount, Class
df_bronze = (
    df_raw
    .withColumnRenamed("Class", "is_fraud")
    .withColumnRenamed("Amount", "amount")
    .withColumnRenamed("Time", "time_seconds")
    # Convert time_seconds to a proper timestamp
    # Dataset starts at 0 seconds -- treat as seconds since start of recording
    .withColumn(
        "transaction_id",
        F.concat(F.lit("txn_"), F.monotonically_increasing_id().cast("string"))
    )
    .withColumn(
        "user_id",
        # Simulate user_id by bucketing -- real dataset has no user_id
        F.concat(
            F.lit("user_"),
            F.lpad((F.col("time_seconds") % 1000).cast("int").cast("string"), 4, "0")
        )
    )
    .withColumn(
        "timestamp",
        # Convert seconds offset to timestamp starting from a base date
        (F.to_timestamp(F.lit("2024-01-01")) + F.expr("INTERVAL 1 SECOND") * F.col("time_seconds"))
    )
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source", F.lit("kaggle_creditcard_fraud"))
)

# COMMAND ----------

# Verify schema
print("Schema:")
df_bronze.printSchema()

print(f"\nTotal records: {df_bronze.count()}")
print(f"Fraud records: {df_bronze.filter(F.col('is_fraud') == 1).count()}")
print(f"Fraud rate: {df_bronze.filter(F.col('is_fraud') == 1).count() / df_bronze.count():.4%}")

display(df_bronze.select(
    "transaction_id", "user_id", "timestamp", "amount", "is_fraud"
).limit(10))

# COMMAND ----------

# Write to Delta bronze table
(
    df_bronze
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TABLE)
)

print(f"Bronze table written: {BRONZE_TABLE}")
print(f"Total records: {df_bronze.count()}")

# COMMAND ----------

# Verify Delta table
df_verify = spark.read.format("delta").table(BRONZE_TABLE)
print(f"Verified records in Delta: {df_verify.count()}")
display(
    df_verify
    .select("transaction_id", "user_id", "timestamp", "amount", "is_fraud")
    .orderBy("timestamp")
    .limit(10)
)

# COMMAND ----------

