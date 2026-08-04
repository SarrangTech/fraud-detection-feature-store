"""Builds the exact feature vector the model expects, from a live transaction plus
whatever this user's precomputed feature hash in Redis contains.

MUST STAY IN SYNC with FEATURE_COLS in notebooks/02_model_training.py, and the proxy
field names must match notebooks/01_feature_store_registration.py's output exactly --
that notebook is the canonical schema for user_fraud_features (dbt's
silver_user_features.sql is the offline/batch equivalent and is schema-tested against
it, see dbt/tests/). The three live in different runtimes so they can't share a single
Python import today; extracting a small shared package installed on both the training
cluster and this service is the natural v2 fix if this list needs to change often.

TRAIN/SERVE APPROXIMATION (documented, not a bug): training uses exact point-in-time
rolling windows computed by dbt (txn_count_1h, amount_stddev_24h, ...). Recomputing
those exactly per request isn't possible in sub-10ms without re-scanning history, so
at serving time we substitute the closest available signal: this user's all-time
rolling aggregates from the Feature Store (user_id primary key), refreshed into Redis
every pipeline run by notebooks/03_redis_feature_sync.py. See docs/architecture.md.
"""
from __future__ import annotations

# Column names must match silver_user_features dbt model and
# 01_feature_store_registration.py output.
FEATURE_COLS = [
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "total_amount_1h", "total_amount_24h", "avg_amount_24h", "max_amount_24h", "min_amount_24h",
    "amount_vs_avg_24h", "amount_vs_max_7d",
    "amount_stddev_24h", "amount_stddev_7d",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V14", "V16", "V17", "V18", "V19", "V21",
    "amount",
]

# Cold-start default: a brand-new user_id with no Redis entry yet (first transaction
# ever, or the Redis sync job hasn't run since they were created). Zeros make the
# rolling-window features look like "no history", which is honest -- there is none.
_COLD_START_PROXY = {
    "avg_velocity_1h": 0.0, "avg_velocity_24h": 0.0, "avg_velocity_7d": 0.0,
    "avg_daily_spend": 0.0, "avg_amount_all_time": 0.0, "max_amount_all_time": 0.0,
    "avg_amount_volatility": 0.0, "amount_stddev_all_time": 0.0,
}


def build_feature_vector(transaction: dict, user_proxy_features: dict | None) -> dict[str, float]:
    proxy = user_proxy_features or _COLD_START_PROXY
    amount = float(transaction["amount"])

    avg_amount_all_time = proxy.get("avg_amount_all_time") or amount
    max_amount_all_time = proxy.get("max_amount_all_time") or amount

    row = {
        "txn_count_1h": proxy.get("avg_velocity_1h", 0.0),
        "txn_count_24h": proxy.get("avg_velocity_24h", 0.0),
        "txn_count_7d": proxy.get("avg_velocity_7d", 0.0),
        "total_amount_1h": amount * proxy.get("avg_velocity_1h", 0.0),
        "total_amount_24h": proxy.get("avg_daily_spend", 0.0),
        "avg_amount_24h": avg_amount_all_time,
        "max_amount_24h": max_amount_all_time,
        # No trailing-minimum equivalent exists in user_fraud_features -- fall back to
        # this transaction's own amount, same as the cold-start case.
        "min_amount_24h": amount,
        "amount_vs_avg_24h": amount / (avg_amount_all_time + 0.01),
        "amount_vs_max_7d": amount / (max_amount_all_time + 0.01),
        "amount_stddev_24h": proxy.get("avg_amount_volatility", 0.0),
        "amount_stddev_7d": proxy.get("amount_stddev_all_time", 0.0),
        "amount": amount,
    }
    for i in list(range(1, 13)) + [14, 16, 17, 18, 19, 21]:
        row[f"V{i}"] = float(transaction.get(f"V{i}", 0.0))

    return {col: row[col] for col in FEATURE_COLS}
