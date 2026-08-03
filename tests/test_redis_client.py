import fakeredis
import pytest

from serving import redis_client


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_client, "get_client", lambda: fake)
    return fake


def test_get_user_features_cache_miss_returns_none():
    assert redis_client.get_user_features("user_unknown") is None


def test_get_user_features_round_trip(fake_redis):
    fake_redis.hset(
        f"{redis_client.settings.redis_feature_key_prefix}:user_00042",
        mapping={"avg_velocity_24h": "3.5", "avg_amount_all_time": "88.20"},
    )

    features = redis_client.get_user_features("user_00042")

    assert features == {"avg_velocity_24h": 3.5, "avg_amount_all_time": 88.20}


def test_ping(fake_redis):
    assert redis_client.ping() is True
