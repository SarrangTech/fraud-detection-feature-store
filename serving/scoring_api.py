"""Real-time fraud scoring service.

Pipeline: transaction arrives -> Redis feature lookup -> model scores -> APPROVE/DECLINE,
with a latency budget of `settings.scoring_latency_budget_ms` (default 100ms) end-to-end.

Run locally:
    uvicorn serving.scoring_api:app --reload --port 8080
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException

from serving import redis_client
from serving.config import settings
from serving.feature_mapping import build_feature_vector
from serving.model_loader import load_model, predict_fraud_probability
from serving.schemas import LatencyBreakdown, ScoringResponse, TransactionRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s scoring_api: %(message)s")
log = logging.getLogger("scoring_api")
audit_log = logging.getLogger("scoring_api.audit")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Preloading model so the first request isn't the one paying for it...")
    load_model()
    if not redis_client.ping():
        log.warning("Redis is not reachable at startup (%s:%s) — cold-start proxy features will be used until it recovers.",
                     settings.redis_host, settings.redis_port)
    if not settings.scoring_api_key:
        log.warning("SCORING_API_KEY is not set -- /score will reject every request until it is configured.")
    yield


app = FastAPI(title="Fraud Detection Real-Time Scoring", lifespan=lifespan)


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Fails closed: an unconfigured SCORING_API_KEY rejects every request rather than
    silently allowing them through, so /score can never be reachable unauthenticated
    by accident (only /health is exempt -- infra health checks need it reachable
    without credentials)."""
    if not settings.scoring_api_key or x_api_key != settings.scoring_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


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


@app.post("/score", response_model=ScoringResponse, dependencies=[Depends(require_api_key)])
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
    except Exception as exc:  # surface as a clean 503, don't leak internals
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
    feature_source = "redis" if user_features is not None else "cold_start"

    # Structured per-decision audit record -- every successful /score call, not just
    # errors. This is what a chargeback/SAR/model-governance review reconstructs from
    # after the fact; a system that only logs exceptions can't answer "what did we
    # decide and on what evidence" for the transactions that scored normally.
    audit_log.info(json.dumps({
        "event": "fraud_decision",
        "transaction_id": transaction.transaction_id,
        "user_id": transaction.user_id,
        "amount": transaction.amount,
        "fraud_probability": round(fraud_probability, 4),
        "decision": decision,
        "threshold": settings.fraud_decision_threshold,
        "feature_source": feature_source,
        "latency_ms": round(total_ms, 1),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }))

    return ScoringResponse(
        transaction_id=transaction.transaction_id,
        user_id=transaction.user_id,
        fraud_probability=round(fraud_probability, 4),
        risk_level=risk_level(fraud_probability),
        decision=decision,
        feature_source=feature_source,
        latency=LatencyBreakdown(
            feature_lookup_ms=round(feature_lookup_ms, 3),
            scoring_ms=round(scoring_ms, 3),
            total_ms=round(total_ms, 3),
        ),
    )
