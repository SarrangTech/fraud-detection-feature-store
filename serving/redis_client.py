"""Thin Redis wrapper for the online feature store.

Uses a module-level connection pool (created once, reused across requests) since
opening a new TCP connection per request would blow the sub-10ms lookup budget on
its own. Callers get a dict of floats back, never raw Redis bytes/strings.
"""
from __future__ import annotations

import redis

from serving.config import settings

_pool = redis.ConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    password=settings.redis_password,
    socket_connect_timeout=2,
    socket_timeout=2,
    decode_responses=True,
)


def get_client() -> redis.Redis:
    return redis.Redis(connection_pool=_pool)


def get_user_features(user_id: str) -> dict[str, float] | None:
    """Fetch the precomputed feature hash for a user. Returns None on a cache miss
    (new/unseen user_id) so callers can fall back to a cold-start default."""
    key = f"{settings.redis_feature_key_prefix}:{user_id}"
    raw = get_client().hgetall(key)
    if not raw:
        return None
    return {k: float(v) for k, v in raw.items()}


def ping() -> bool:
    try:
        return get_client().ping()
    except redis.RedisError:
        return False
