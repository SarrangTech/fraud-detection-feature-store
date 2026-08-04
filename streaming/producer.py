"""
Kafka producer that replays the Kaggle Credit Card Fraud Detection CSV
(creditcard.csv) row by row, simulating a real-time transaction stream.

The source dataset has no customer/account identifier (see data/README.md),
so each row is deterministically hashed into a fixed pool of simulated user
accounts here, at the point of ingestion, before anything downstream ever
sees the record.

Usage:
    python streaming/producer.py
    python streaming/producer.py --events-per-second 100 --limit 5000
    python streaming/producer.py --loop                      # replay forever
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import KafkaError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s producer: %(message)s",
)
log = logging.getLogger("producer")

# Any fixed anchor date works — only relative offsets between transactions matter
# downstream (rolling windows, velocity). Kept stable so re-runs are comparable.
DATASET_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ProducerConfig:
    bootstrap_servers: str
    topic: str
    client_id: str
    raw_data_path: Path
    events_per_second: float
    user_pool_size: int
    limit: int | None
    loop: bool

    @classmethod
    def from_env(cls, args: argparse.Namespace) -> ProducerConfig:
        return cls(
            bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            topic=os.environ.get("KAFKA_TOPIC_TRANSACTIONS", "fraud.transactions.raw"),
            client_id=os.environ.get("KAFKA_PRODUCER_CLIENT_ID", "creditcard-producer"),
            raw_data_path=Path(args.data_path or os.environ.get("RAW_DATA_PATH", "data/raw/creditcard.csv")),
            events_per_second=args.events_per_second
            if args.events_per_second is not None
            else float(os.environ.get("PRODUCER_EVENTS_PER_SECOND", "20")),
            user_pool_size=int(os.environ.get("SIMULATED_USER_POOL_SIZE", "5000")),
            limit=args.limit,
            loop=args.loop,
        )


def simulate_user_id(row_index: int, time_offset_seconds: float, pool_size: int) -> str:
    """Deterministically hash a transaction into one of `pool_size` simulated accounts.

    Uses the row's original CSV position plus its Time offset so the same input
    file always maps to the same user_id sequence, which keeps rolling-window
    features reproducible across producer restarts and re-runs of the pipeline.
    """
    digest = hashlib.md5(f"{row_index}:{time_offset_seconds}".encode()).hexdigest()
    bucket = int(digest, 16) % pool_size
    return f"user_{bucket:05d}"


def read_transactions(path: Path, limit: int | None) -> Iterator[tuple[int, dict]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                return
            yield idx, row


def to_event(row_index: int, row: dict, pool_size: int) -> dict:
    time_offset = float(row["Time"])
    user_id = simulate_user_id(row_index, time_offset, pool_size)
    event_time = DATASET_EPOCH + timedelta(seconds=time_offset)

    event = {
        "transaction_id": f"txn_{row_index:08d}",
        "user_id": user_id,
        "event_time": event_time.isoformat(),
        "time_offset_seconds": time_offset,
        "amount": float(row["Amount"]),
        "is_fraud": int(row["Class"]),
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }
    for i in range(1, 29):
        event[f"V{i}"] = float(row[f"V{i}"])
    return event


class GracefulShutdown:
    def __init__(self) -> None:
        self.should_stop = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame) -> None:
        log.info("Shutdown signal received (%s) — draining producer...", signum)
        self.should_stop = True


def run(config: ProducerConfig) -> None:
    if not config.raw_data_path.exists():
        log.error(
            "Raw dataset not found at %s. Run `make download-data` first (see data/README.md).",
            config.raw_data_path,
        )
        sys.exit(1)

    producer = KafkaProducer(
        bootstrap_servers=config.bootstrap_servers,
        client_id=config.client_id,
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=5,
        linger_ms=5,
    )

    interval = 0.0 if config.events_per_second <= 0 else 1.0 / config.events_per_second
    shutdown = GracefulShutdown()
    sent, fraud_sent, errors = 0, 0, 0
    started_at = time.monotonic()
    last_log_at = started_at

    log.info(
        "Streaming %s -> topic '%s' on %s (%.1f events/sec, user pool=%d)",
        config.raw_data_path,
        config.topic,
        config.bootstrap_servers,
        config.events_per_second,
        config.user_pool_size,
    )

    def send_one(row_index: int, row: dict) -> None:
        nonlocal sent, fraud_sent, errors
        event = to_event(row_index, row, config.user_pool_size)
        try:
            producer.send(config.topic, key=event["user_id"], value=event)
        except KafkaError:
            errors += 1
            log.exception("Failed to enqueue transaction_id=%s", event["transaction_id"])
            return
        sent += 1
        if event["is_fraud"]:
            fraud_sent += 1

    try:
        while True:
            for row_index, row in read_transactions(config.raw_data_path, config.limit):
                if shutdown.should_stop:
                    break
                send_one(row_index, row)

                now = time.monotonic()
                if now - last_log_at >= 5:
                    elapsed = now - started_at
                    log.info(
                        "sent=%d fraud=%d errors=%d rate=%.1f/s",
                        sent, fraud_sent, errors, sent / elapsed if elapsed else 0,
                    )
                    last_log_at = now

                if interval:
                    time.sleep(interval)

            if not config.loop or shutdown.should_stop:
                break
            log.info("Reached end of dataset — looping (--loop set)")
    finally:
        producer.flush(timeout=30)
        producer.close(timeout=30)
        log.info("Producer stopped. Total sent=%d fraud=%d errors=%d", sent, fraud_sent, errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", help="Path to creditcard.csv (default: $RAW_DATA_PATH)")
    parser.add_argument("--events-per-second", type=float, default=None, help="Throttle rate; 0 = unthrottled")
    parser.add_argument("--limit", type=int, default=None, help="Only stream the first N rows (useful for smoke tests)")
    parser.add_argument("--loop", action="store_true", help="Replay the dataset indefinitely")
    return parser.parse_args()


if __name__ == "__main__":
    run(ProducerConfig.from_env(parse_args()))
