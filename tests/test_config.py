"""Guards against config fields silently drifting from .env.example -- caught a real
bug during evidence-gathering where scripts/seed_redis.py referenced
settings.redis_feature_ttl_seconds but Settings never declared that field."""
from serving.config import Settings


def test_settings_declares_every_redis_field_env_example_documents():
    settings = Settings()
    for field in ("redis_host", "redis_port", "redis_db", "redis_feature_key_prefix", "redis_feature_ttl_seconds"):
        assert hasattr(settings, field), f"Settings is missing '{field}' (see .env.example)"
