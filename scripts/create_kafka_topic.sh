#!/usr/bin/env bash
# Creates the transactions topic with partitioning/retention suited to this workload.
# Partition key is user_id (see streaming/producer.py), so per-user ordering is preserved
# within a partition, which matters for correct rolling-window feature computation.
set -euo pipefail

# On Windows/Git Bash, MSYS rewrites POSIX-looking arguments (e.g. /opt/kafka/...)
# into Windows paths before they reach `docker exec`, breaking the in-container path.
# Harmless no-op on Linux/macOS.
export MSYS_NO_PATHCONV=1

: "${KAFKA_TOPIC_TRANSACTIONS:=fraud.transactions.raw}"
: "${KAFKA_BOOTSTRAP_SERVERS:=localhost:9092}"
: "${KAFKA_TOPIC_PARTITIONS:=6}"
: "${KAFKA_TOPIC_RETENTION_MS:=604800000}" # 7 days

echo "Creating topic '$KAFKA_TOPIC_TRANSACTIONS' on $KAFKA_BOOTSTRAP_SERVERS ..."

docker exec fraud-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
  --create --if-not-exists \
  --topic "$KAFKA_TOPIC_TRANSACTIONS" \
  --partitions "$KAFKA_TOPIC_PARTITIONS" \
  --replication-factor 1 \
  --config retention.ms="$KAFKA_TOPIC_RETENTION_MS"

docker exec fraud-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
  --describe --topic "$KAFKA_TOPIC_TRANSACTIONS"
