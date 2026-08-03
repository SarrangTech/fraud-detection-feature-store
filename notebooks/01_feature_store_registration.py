# Databricks notebook source
# TITLE: 01_feature_store_registration
# PURPOSE: Register the dbt-built silver_user_features table (primary key: user_id) as a
#          Databricks Feature Engineering table, so it can be joined at training time via
#          FeatureLookup and synced to Redis for online serving.
# RUN AFTER: `dbt run` has materialized dbt.silver_user_features (see dbt/models/silver/).
# PROJECT: Fraud Detection Feature Store

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("uc_catalog", "fraud_detection")
dbutils.widgets.text("uc_schema", "feature_store")

CATALOG = dbutils.widgets.get("uc_catalog")
SCHEMA = dbutils.widgets.get("uc_schema")

# dbt materializes into <catalog>.<schema>.silver_user_features per dbt/dbt_project.yml
SILVER_USER_TABLE = f"{CATALOG}.{SCHEMA}.silver_user_features"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.user_fraud_features"

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# COMMAND ----------

df_user_features = spark.read.table(SILVER_USER_TABLE)
print(f"Users: {df_user_features.count()}, columns: {len(df_user_features.columns)}")
display(df_user_features.limit(5))

# COMMAND ----------

# Idempotent create-or-update: first run creates the Feature Store table, every
# subsequent run (e.g. scheduled after each `dbt run`) merges in refreshed features.
existing_tables = [t.name for t in spark.catalog.listTables(f"{CATALOG}.{SCHEMA}")]

if "user_fraud_features" not in existing_tables:
    fe.create_table(
        name=FEATURE_TABLE,
        primary_keys=["user_id"],
        df=df_user_features,
        description=(
            "User-level fraud detection features (velocity, spend patterns, "
            "volatility) computed by dbt from streaming transaction data. "
            "Primary key: user_id. Refreshed after every dbt run."
        ),
    )
    print(f"Feature table CREATED: {FEATURE_TABLE}")
else:
    fe.write_table(name=FEATURE_TABLE, df=df_user_features, mode="merge")
    print(f"Feature table UPDATED (merge): {FEATURE_TABLE}")

# COMMAND ----------

df_verify = fe.read_table(name=FEATURE_TABLE)
print(f"Feature table row count: {df_verify.count()}")
display(df_verify.limit(10))
