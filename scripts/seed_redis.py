"""Seeds Redis with a handful of realistic precomputed user feature hashes, so
serving/scoring_api.py has something to look up without waiting on a full
Databricks Feature Store -> Redis sync (notebooks/03_redis_feature_sync.py) run.

Useful for local demos / smoke-testing the real-time scoring pipeline end-to-end.

Usage:
    python scripts/seed_redis.py
"""
from __future__ import annotations

import redis

from serving.config import settings

# user_00042: an established, modest-spending account. A $1,250 transaction against
# this profile is ~16x their historical average and ~3x their historical max, which
# is exactly the kind of spend-pattern spike the model is trained to flag.
DEMO_USERS = {
    "user_00042": {
        "avg_velocity_1h": 0.4,
        "avg_velocity_24h": 3.0,
        "avg_velocity_7d": 18.0,
        "avg_daily_spend": 210.0,
        "avg_amount_all_time": 76.50,
        "max_amount_all_time": 410.00,
        "avg_amount_volatility": 24.10,
        "amount_stddev_all_time": 38.75,
    },
    "user_00099": {  # frequent, high-velocity spender -- a $1,250 charge is unremarkable
        "avg_velocity_1h": 2.5,
        "avg_velocity_24h": 14.0,
        "avg_velocity_7d": 90.0,
        "avg_daily_spend": 1800.0,
        "avg_amount_all_time": 640.00,
        "max_amount_all_time": 2100.00,
        "avg_amount_volatility": 310.0,
        "amount_stddev_all_time": 290.0,
    },
    "user_00777": {  # low-activity, low-spend account
        "avg_velocity_1h": 0.05,
        "avg_velocity_24h": 0.4,
        "avg_velocity_7d": 2.0,
        "avg_daily_spend": 35.0,
        "avg_amount_all_time": 28.00,
        "max_amount_all_time": 95.00,
        "avg_amount_volatility": 9.5,
        "amount_stddev_all_time": 12.0,
    },
}


def main() -> None:
    client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        db=settings.redis_db,
        decode_responses=True,
    )
    client.ping()

    for user_id, features in DEMO_USERS.items():
        key = f"{settings.redis_feature_key_prefix}:{user_id}"
        client.hset(key, mapping={k: str(v) for k, v in features.items()})
        client.expire(key, settings.redis_feature_ttl_seconds)
        print(f"seeded {key} ({len(features)} fields, ttl={settings.redis_feature_ttl_seconds}s)")

    print(f"\nDone. Seeded {len(DEMO_USERS)} users into Redis @ {settings.redis_host}:{settings.redis_port}")


if __name__ == "__main__":
    main()
