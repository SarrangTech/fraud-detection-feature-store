from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Redis (online feature store)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_feature_key_prefix: str = "user_features"

    # MLflow / Unity Catalog model registry
    mlflow_tracking_uri: str = "databricks"
    mlflow_registry_uri: str = "databricks-uc"
    uc_model_name: str = "fraud_detection.feature_store.fraud_gbm_classifier"
    model_version_or_alias: str = "1"
    # Set to a local path (e.g. a directory produced by mlflow.sklearn.save_model) to
    # score with a local model instead of hitting a Databricks workspace -- used for
    # local dev / smoke testing without a real Unity Catalog registry.
    model_local_path: str | None = None

    # Scoring behavior
    fraud_decision_threshold: float = 0.30
    scoring_latency_budget_ms: float = 100.0

    # API
    scoring_api_host: str = "0.0.0.0"
    scoring_api_port: int = 8080


settings = Settings()
