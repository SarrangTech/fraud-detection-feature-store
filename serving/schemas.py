from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float = Field(gt=0)
    # PCA-anonymized features from the source dataset; unused ones default to 0.0
    # so callers only need to send the subset the model actually consumes.
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V14: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V21: float = 0.0


class LatencyBreakdown(BaseModel):
    feature_lookup_ms: float
    scoring_ms: float
    total_ms: float


class ScoringResponse(BaseModel):
    transaction_id: str
    user_id: str
    fraud_probability: float
    risk_level: str
    decision: str
    feature_source: str  # "redis" | "cold_start"
    latency: LatencyBreakdown
