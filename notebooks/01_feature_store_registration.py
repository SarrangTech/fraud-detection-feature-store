# Databricks notebook source
# TITLE: 01_feature_store_registration
# PURPOSE: Register a user-level (PK: user_id) feature table into the Databricks
#          Feature Store, so it can be synced to Redis for online serving.
#
# WORKSPACE NOTE: this workspace's silver table is `silver_features` -- one row per
# TRANSACTION (284,807 rows), not a per-user aggregate. The Feature Store table
# this notebook targets, `user_fraud_features`, already exists from a previous
# session (1,000 rows) -- if so, this notebook just reuses it as-is rather than
# rebuilding it. It only aggregates `silver_features` by user_id and creates the
# table if `user_fraud_features` doesn't exist yet. See WORKSPACE_SETUP.md.
#
# CANONICAL SCHEMA: the column names produced here are the source of truth for
# user_fraud_features -- this is the path that actually runs against the live
# workspace. dbt/models/silver/silver_user_features.sql and
# serving/feature_mapping.py must match these names exactly (see the dbt schema
# test in dbt/tests/ and README Known Limitations).
# PROJECT: Fraud Detection Feature Store

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("uc_catalog", "workspace")
dbutils.widgets.text("uc_schema", "fraud_feature_store")

CATALOG = dbutils.widgets.get("uc_catalog")
SCHEMA = dbutils.widgets.get("uc_schema")

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_features"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.user_fraud_features"

from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# COMMAND ----------

existing_tables = [t.name for t in spark.catalog.listTables(f"{CATALOG}.{SCHEMA}")]

if "user_fraud_features" in existing_tables:
    # Already built (this workspace has it from a previous session) -- reuse it
    # rather than recompute, per the "don't recreate what already exists" policy.
    df_user_features = fe.read_table(name=FEATURE_TABLE)
    print(f"{FEATURE_TABLE} already exists -- reusing it as-is ({df_user_features.count()} users).")
else:
    # Build it by aggregating the per-transaction silver_features table by user_id.
    # Column names match what originally created this table (see WORKSPACE_SETUP.md)
    # so a table built fresh here has the same schema either way.
    df_silver = spark.read.table(SILVER_TABLE)
    df_user_features = (
        df_silver.groupBy("user_id").agg(
            F.count("transaction_id").alias("total_txn_count"),
            F.max("timestamp").alias("last_transaction_time"),
            F.avg("amount").alias("avg_amount_all_time"),
            F.max("amount").alias("max_amount_all_time"),
            F.stddev("amount").alias("amount_stddev_all_time"),
            F.sum("amount").alias("total_spend_all_time"),
            F.avg("txn_count_1h").alias("avg_velocity_1h"),
            F.avg("txn_count_24h").alias("avg_velocity_24h"),
            F.avg("txn_count_7d").alias("avg_velocity_7d"),
            F.max("txn_count_1h").alias("max_velocity_1h"),
            F.avg("total_amount_24h").alias("avg_daily_spend"),
            F.avg("amount_stddev_24h").alias("avg_amount_volatility"),
            F.avg("amount_vs_avg_24h").alias("avg_amount_ratio"),
            F.avg("V1").alias("avg_v1"),
            F.avg("V2").alias("avg_v2"),
            F.avg("V3").alias("avg_v3"),
            F.avg("V4").alias("avg_v4"),
            F.avg("V5").alias("avg_v5"),
        ).fillna(0)
    )
    print(f"Built user-level aggregate from {SILVER_TABLE}: {df_user_features.count()} users.")

print(f"Columns: {df_user_features.columns}")
display(df_user_features.limit(5))

# COMMAND ----------

# Idempotent create-only: if the table already existed above, nothing is written
# here -- this notebook never overwrites an existing Feature Store table.
if "user_fraud_features" not in existing_tables:
    fe.create_table(
        name=FEATURE_TABLE,
        primary_keys=["user_id"],
        df=df_user_features,
        description=(
            "User-level fraud detection features (velocity, spend patterns, "
            "volatility) aggregated from silver_features. Primary key: user_id."
        ),
    )
    print(f"Feature table CREATED: {FEATURE_TABLE}")
else:
    print("Feature table already existed -- nothing written (see cell above).")

# COMMAND ----------

df_verify = fe.read_table(name=FEATURE_TABLE)
print(f"Feature table row count: {df_verify.count()}")
display(df_verify.limit(10))
