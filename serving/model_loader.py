"""Loads the trained fraud model exactly once at process startup and keeps it in
memory for the life of the process -- re-loading from MLflow/Unity Catalog on every
request would alone blow the 100ms latency budget.
"""
from __future__ import annotations

import functools
import logging

import pandas as pd

from serving.config import settings

log = logging.getLogger("model_loader")


@functools.lru_cache(maxsize=1)
def load_model():
    import mlflow.sklearn

    if settings.model_local_path:
        log.info("Loading model from local path: %s", settings.model_local_path)
        return mlflow.sklearn.load_model(settings.model_local_path)

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)
    model_uri = f"models:/{settings.uc_model_name}/{settings.model_version_or_alias}"
    log.info("Loading model from Unity Catalog registry: %s", model_uri)
    # mlflow.sklearn.load_model (native flavor, not pyfunc) returns the raw
    # GradientBoostingClassifier so predict_proba is available for scoring.
    return mlflow.sklearn.load_model(model_uri)


def predict_fraud_probability(model, feature_row: dict[str, float]) -> float:
    df = pd.DataFrame([feature_row])
    return float(model.predict_proba(df)[:, 1][0])
