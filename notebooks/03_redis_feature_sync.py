# Databricks notebook source
# TITLE: 03_redis_feature_sync
# PURPOSE: Push the Databricks Feature Store's user_fraud_features table (PK: user_id)
#          into Redis as one hash per user, so serving/scoring_api.py can do a single
#          sub-10ms HGETALL instead of a Spark/Delta read on every scoring request.
# RUN: on a schedule (see databricks/resources/jobs.yml) after 01_feature_store_registration.
# PROJECT: Fraud Detection Feature Store

# COMMAND ----------

# MAGIC %pip install redis
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("uc_catalog", "fraud_detection")
dbutils.widgets.text("uc_schema", "feature_store")
dbutils.widgets.text("redis_host", "localhost")
dbutils.widgets.text("redis_port", "6379")
dbutils.widgets.text("redis_key_prefix", "user_features")
dbutils.widgets.text("redis_ttl_seconds", "86400")

CATALOG = dbutils.widgets.get("uc_catalog")
SCHEMA = dbutils.widgets.get("uc_schema")
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.user_fraud_features"

REDIS_HOST = dbutils.widgets.get("redis_host")
REDIS_PORT = int(dbutils.widgets.get("redis_port"))
REDIS_PASSWORD = dbutils.secrets.get(scope="fraud-detection", key="redis-password") if dbutils.secrets.list("fraud-detection") else None
KEY_PREFIX = dbutils.widgets.get("redis_key_prefix")
TTL_SECONDS = int(dbutils.widgets.get("redis_ttl_seconds"))

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# COMMAND ----------

df_features = fe.read_table(name=FEATURE_TABLE)
feature_cols = [c for c in df_features.columns if c != "user_id"]
print(f"Syncing {df_features.count()} users x {len(feature_cols)} features to Redis @ {REDIS_HOST}:{REDIS_PORT}")

# COMMAND ----------

def sync_partition(rows):
    """Runs on each Spark executor: one Redis connection per partition, pipelined writes."""
    import redis

    client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=0,
        socket_connect_timeout=5, socket_timeout=5,
    )
    pipe = client.pipeline(transaction=False)
    batch = 0

    for row in rows:
        d = row.asDict()
        user_id = d.pop("user_id")
        # Redis hash fields must be strings; None -> "0" so serving-side parsing is uniform.
        mapping = {k: str(v) if v is not None else "0" for k, v in d.items()}
        key = f"{KEY_PREFIX}:{user_id}"
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, TTL_SECONDS)
        batch += 1
        if batch >= 500:
            pipe.execute()
            batch = 0

    if batch:
        pipe.execute()
    client.close()


df_features.foreachPartition(sync_partition)
print(f"Redis sync complete. TTL={TTL_SECONDS}s, key pattern='{KEY_PREFIX}:<user_id>'")

# COMMAND ----------

# Spot-check one user round-trip
import redis

_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=0)
sample_user = df_features.select("user_id").limit(1).collect()[0]["user_id"]
print(f"Sample key '{KEY_PREFIX}:{sample_user}':")
print(_client.hgetall(f"{KEY_PREFIX}:{sample_user}"))
_client.close()
