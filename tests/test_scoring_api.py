import fakeredis
import pytest
from fastapi.testclient import TestClient

from serving import redis_client, scoring_api

TEST_API_KEY = "test-key-for-pytest"
AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


class _FakeModel:
    """predict_proba returns a high fraud score for large amounts, low otherwise --
    enough behavior to exercise the APPROVE/DECLINE branch without a real model."""

    def predict_proba(self, df):
        import numpy as np

        amounts = df["amount"].to_numpy()
        proba_fraud = np.where(amounts > 1000, 0.95, 0.05)
        return np.column_stack([1 - proba_fraud, proba_fraud])


@pytest.fixture(autouse=True)
def patched_dependencies(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_client, "get_client", lambda: fake)
    monkeypatch.setattr(scoring_api, "load_model", lambda: _FakeModel())
    monkeypatch.setattr(scoring_api.settings, "scoring_api_key", TEST_API_KEY)
    return fake


@pytest.fixture
def client():
    with TestClient(scoring_api.app) as c:
        yield c


def test_health_does_not_require_api_key(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["redis_reachable"] is True


def test_score_rejects_missing_api_key(client):
    resp = client.post("/score", json={"transaction_id": "txn_0", "user_id": "user_0", "amount": 10.0})
    assert resp.status_code == 401


def test_score_rejects_wrong_api_key(client):
    resp = client.post(
        "/score",
        json={"transaction_id": "txn_0", "user_id": "user_0", "amount": 10.0},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_score_approves_small_transaction(client):
    resp = client.post(
        "/score", json={"transaction_id": "txn_1", "user_id": "user_1", "amount": 45.0}, headers=AUTH_HEADERS
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "APPROVE"
    assert body["feature_source"] == "cold_start"
    assert body["latency"]["total_ms"] > 0


def test_score_declines_large_transaction(client):
    resp = client.post(
        "/score", json={"transaction_id": "txn_2", "user_id": "user_2", "amount": 5000.0}, headers=AUTH_HEADERS
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "DECLINE"
    assert body["risk_level"] == "HIGH"


def test_score_uses_redis_features_when_present(client, patched_dependencies):
    patched_dependencies.hset(
        f"{redis_client.settings.redis_feature_key_prefix}:user_00042",
        mapping={"avg_amount_all_time": "60.0"},
    )
    resp = client.post(
        "/score", json={"transaction_id": "txn_3", "user_id": "user_00042", "amount": 65.0}, headers=AUTH_HEADERS
    )
    assert resp.json()["feature_source"] == "redis"


def test_score_rejects_non_positive_amount(client):
    resp = client.post(
        "/score", json={"transaction_id": "txn_4", "user_id": "user_4", "amount": 0}, headers=AUTH_HEADERS
    )
    assert resp.status_code == 422
