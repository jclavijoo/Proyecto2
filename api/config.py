"""Configuración de la API (variables de entorno desde ConfigMap/Secret)."""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name: str = os.getenv("MODEL_NAME", "diabetes-readmission")
    model_alias: str = os.getenv("MODEL_ALIAS", "champion")
    # Cada cuántos segundos se consulta si el alias apunta a otra versión
    model_refresh_seconds: int = int(os.getenv("MODEL_REFRESH_SECONDS", "60"))
    decision_threshold: float = float(os.getenv("DECISION_THRESHOLD", "0.5"))
    log_inferences: bool = os.getenv("LOG_INFERENCES", "true").lower() == "true"
    db: dict = field(default_factory=lambda: {
        "host": os.getenv("DATA_DB_HOST", "postgres-data"),
        "port": int(os.getenv("DATA_DB_PORT", "5432")),
        "dbname": os.getenv("DATA_DB_NAME", "diabetes"),
        "user": os.getenv("DATA_DB_USER", "api_writer"),
        "password": os.getenv("DATA_DB_PASSWORD", ""),
    })
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    api_version: str = os.getenv("API_VERSION", "1.0.0")


settings = Settings()
