"""Real-time fraud scoring service.

Pipeline: transaction arrives -> Redis feature lookup -> model scores -> APPROVE/DECLINE,
with a latency budget of `settings.scoring_latency_budget_ms` (default 100ms) end-to-end.

Run locally:
    uvicorn serving.scoring_api:app --reload --port 8080
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from serving import redis_client
from serving.config import settings
from serving.feature_mapping import build_feature_vector
from serving.model_loader import load_model, predict_fraud_probability
from serving.schemas import LatencyBreakdown, ScoringResponse, TransactionRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s scoring_api: %(message)s")
log = logging.getLogger("scoring_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Preloading model so the first request isn't the one paying for it...")
    load_model()
    if not redis_client.ping():
        log.warning("Redis is not reachable at startup (%s:%s) — cold-start proxy features will be used until it recovers.",
                     settings.redis_host, settings.redis_port)
    yield


app = FastAPI(title="Fraud Detection Real-Time Scoring", lifespan=lifespan)


def risk_level(probability: float) -> str:
    if probability >= 0.5:
        return "HIGH"
    if probability >= settings.fraud_decision_threshold:
        return "MEDIUM"
    return "LOW"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "redis_reachable": redis_client.ping(),
        "fraud_decision_threshold": settings.fraud_decision_threshold,
        "latency_budget_ms": settings.scoring_latency_budget_ms,
    }


@app.post("/score", response_model=ScoringResponse)
def score(transaction: TransactionRequest) -> ScoringResponse:
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    user_features = redis_client.get_user_features(transaction.user_id)
    feature_lookup_ms = (time.perf_counter() - t0) * 1000

    feature_vector = build_feature_vector(transaction.model_dump(), user_features)

    t1 = time.perf_counter()
    try:
        model = load_model()
        fraud_probability = predict_fraud_probability(model, feature_vector)
    except Exception as exc:  # noqa: BLE001 - surface as a clean 503, don't leak internals
        log.exception("Model scoring failed for transaction_id=%s", transaction.transaction_id)
        raise HTTPException(status_code=503, detail="Scoring model unavailable") from exc
    scoring_ms = (time.perf_counter() - t1) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000
    if total_ms > settings.scoring_latency_budget_ms:
        log.warning(
            "Latency budget exceeded: %.1fms > %.1fms for transaction_id=%s",
            total_ms, settings.scoring_latency_budget_ms, transaction.transaction_id,
        )

    decision = "DECLINE" if fraud_probability >= settings.fraud_decision_threshold else "APPROVE"

    return ScoringResponse(
        transaction_id=transaction.transaction_id,
        user_id=transaction.user_id,
        fraud_probability=round(fraud_probability, 4),
        risk_level=risk_level(fraud_probability),
        decision=decision,
        feature_source="redis" if user_features is not None else "cold_start",
        latency=LatencyBreakdown(
            feature_lookup_ms=round(feature_lookup_ms, 3),
            scoring_ms=round(scoring_ms, 3),
            total_ms=round(total_ms, 3),
        ),
    )
