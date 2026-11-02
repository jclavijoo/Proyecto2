"""Registro de cada inferencia en inference.prediction_logs (PostgreSQL)."""
from __future__ import annotations

import json
import logging
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import settings
from metrics import DB_LOG_FAILURES

log = logging.getLogger(__name__)
_engine: Engine | None = None

INSERT_SQL = text("""
    INSERT INTO inference.prediction_logs (
        request_id, requested_at, input_payload, features, prediction,
        prediction_label, probability, decision_threshold, model_name,
        model_version, model_alias, model_run_id, latency_ms, status,
        error_message, client_host)
    VALUES (
        :request_id, :requested_at, CAST(:input_payload AS JSONB),
        CAST(:features AS JSONB), :prediction, :prediction_label, :probability,
        :decision_threshold, :model_name, :model_version, :model_alias,
        :model_run_id, :latency_ms, :status, :error_message, :client_host)
""")


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        c = settings.db
        url = (f"postgresql+psycopg2://{quote_plus(c['user'])}:{quote_plus(c['password'])}"
               f"@{c['host']}:{c['port']}/{c['dbname']}")
        _engine = create_engine(
            url, pool_size=settings.db_pool_size, max_overflow=settings.db_max_overflow,
            pool_pre_ping=True, pool_timeout=10, pool_recycle=1800)
    return _engine


def log_inference(record: dict) -> None:
    """Se ejecuta como BackgroundTask: no suma latencia a la respuesta HTTP."""
    if not settings.log_inferences:
        return
    try:
        params = dict(record)
        params["input_payload"] = json.dumps(record["input_payload"], default=str)
        params["features"] = json.dumps(record.get("features"), default=str)
        with get_engine().begin() as conn:
            conn.execute(INSERT_SQL, params)
    except Exception:  # noqa: BLE001 - nunca romper la API por el log
        DB_LOG_FAILURES.inc()
        log.exception("No se pudo registrar la inferencia %s", record.get("request_id"))


def ping() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False
