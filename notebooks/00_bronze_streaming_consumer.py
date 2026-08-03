# Databricks notebook source
# TITLE: 00_bronze_streaming_consumer
# PURPOSE: Spark Structured Streaming job that reads raw transaction events from the
#          Kafka topic populated by streaming/producer.py and writes them, append-only,
#          into a Delta bronze table in Unity Catalog.
# RUN AS: a Databricks Job (see databricks/resources/jobs.yml), continuous or scheduled.
# PROJECT: Fraud Detection Feature Store

# COMMAND ----------

dbutils.widgets.text("uc_catalog", "fraud_detection")
dbutils.widgets.text("uc_schema", "feature_store")
dbutils.widgets.text("kafka_bootstrap_servers", "localhost:9092")
dbutils.widgets.text("kafka_topic", "fraud.transactions.raw")
dbutils.widgets.text("trigger_interval", "10 seconds")
dbutils.widgets.dropdown("run_mode", "continuous", ["continuous", "trigger_once"])

CATALOG = dbutils.widgets.get("uc_catalog")
SCHEMA = dbutils.widgets.get("uc_schema")
KAFKA_BOOTSTRAP_SERVERS = dbutils.widgets.get("kafka_bootstrap_servers")
KAFKA_TOPIC = dbutils.widgets.get("kafka_topic")
TRIGGER_INTERVAL = dbutils.widgets.get("trigger_interval")
RUN_MODE = dbutils.widgets.get("run_mode")

BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_transactions"
CHECKPOINT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/bronze_transactions"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)

# Must match the event shape produced by streaming/producer.py::to_event
transaction_schema = StructType(
    [
        StructField("transaction_id", StringType()),
        StructField("user_id", StringType()),
        StructField("event_time", StringType()),
        StructField("time_offset_seconds", DoubleType()),
        StructField("amount", DoubleType()),
        StructField("is_fraud", IntegerType()),
        StructField("produced_at", StringType()),
    ]
    + [StructField(f"V{i}", DoubleType()) for i in range(1, 29)]
)

# COMMAND ----------

# Read raw bytes from Kafka. `startingOffsets=earliest` only applies the first time a
# checkpoint is created for this query -- subsequent restarts resume from the checkpoint,
# so replaying the producer with --loop will not create duplicate bronze rows on restart.
df_kafka = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "earliest")
    .option("failOnDataLoss", "false")
    .load()
)

df_parsed = (
    df_kafka.select(
        F.col("key").cast("string").alias("kafka_key"),
        F.col("partition").alias("_kafka_partition"),
        F.col("offset").alias("_kafka_offset"),
        F.col("timestamp").alias("_kafka_timestamp"),
        F.from_json(F.col("value").cast("string"), transaction_schema).alias("event"),
    )
    .select("kafka_key", "_kafka_partition", "_kafka_offset", "_kafka_timestamp", "event.*")
    .withColumn("event_time", F.to_timestamp("event_time"))
    .withColumn("produced_at", F.to_timestamp("produced_at"))
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source", F.lit("kafka:" + KAFKA_TOPIC))
)

# COMMAND ----------

write_query = (
    df_parsed.writeStream.format("delta")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .outputMode("append")
    .queryName("bronze_transactions_ingest")
)

if RUN_MODE == "trigger_once":
    # Process everything currently in the topic then stop -- the pattern to use when this
    # notebook runs as a scheduled (rather than always-on) Databricks Job.
    write_query = write_query.trigger(availableNow=True)
else:
    write_query = write_query.trigger(processingTime=TRIGGER_INTERVAL)

query = write_query.toTable(BRONZE_TABLE)

if RUN_MODE == "trigger_once":
    query.awaitTermination()
    print(f"Bronze ingestion (trigger_once) complete -> {BRONZE_TABLE}")
else:
    print(f"Bronze ingestion streaming continuously -> {BRONZE_TABLE} (query id: {query.id})")
    # In an interactive notebook this call blocks the cell; when run as a Job task with
    # run_mode=continuous, the Job itself keeps the cluster/query alive.
    query.awaitTermination()

# COMMAND ----------

# Sanity check (run manually / in trigger_once mode after the query stops)
df_check = spark.read.table(BRONZE_TABLE)
print(f"Bronze row count: {df_check.count()}")
display(
    df_check.select("transaction_id", "user_id", "event_time", "amount", "is_fraud")
    .orderBy(F.col("event_time").desc())
    .limit(10)
)
