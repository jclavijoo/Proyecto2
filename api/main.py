"""API de inferencia - readmisión hospitalaria (<30 días) de pacientes diabéticos."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import db
from config import settings
from metrics import (MODEL_LATENCY, PREDICTED_PROBABILITY, PREDICTION_ERRORS,
                         PREDICTIONS, PrometheusMiddleware)
from model_manager import manager
from schemas import ModelInfo, PatientInput, PredictionResponse
from features import TRANSFORM_VERSION, build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("inference-api")

LABELS = {1: "readmitted_<30", 0: "not_readmitted_<30"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # La carga inicial no bloquea el arranque si MLflow aún no tiene modelo:
    # la API responde 503 en /predict hasta que exista un @champion.
    manager.refresh()
    manager.start_background_refresh()
    yield
    manager.stop()


app = FastAPI(
    title="Diabetes Readmission - API de inferencia",
    version=settings.api_version,
    description="Usa el modelo productivo registrado en MLflow "
                f"(models:/{settings.model_name}@{settings.model_alias}).",
    lifespan=lifespan,
)
app.add_middleware(PrometheusMiddleware)


@app.get("/health")
def health() -> dict:
    current = manager.current
    return {
        "status": "ok",
        "model_loaded": current is not None,
        "model_version": current.version if current else None,
        "transform_version": TRANSFORM_VERSION,
    }


@app.get("/health/db")
def health_db() -> dict:
    ok = db.ping()
    if not ok:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")
    return {"database": "ok"}


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    current = manager.current
    return ModelInfo(
        status="loaded" if current else "not_loaded",
        model_name=settings.model_name,
        model_alias=settings.model_alias,
        model_version=current.version if current else None,
        run_id=current.run_id if current else None,
        loaded_at=current.loaded_at if current else None,
        last_check_at=manager.last_check_at,
        last_error=manager.last_error,
        refresh_interval_seconds=settings.model_refresh_seconds,
        version_tags=current.tags if current else {},
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
    )


@app.post("/reload-model")
def reload_model() -> dict:
    changed = manager.refresh()
    current = manager.current
    return {"reloaded": changed, "model_version": current.version if current else None,
            "last_error": manager.last_error}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientInput, request: Request,
            background_tasks: BackgroundTasks) -> PredictionResponse:
    start = time.perf_counter()
    requested_at = datetime.now(timezone.utc)
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    raw = payload.model_dump(by_alias=True)
    current = manager.current  # referencia local: inmune al hot-swap

    base_log = {
        "request_id": request_id, "requested_at": requested_at, "input_payload": raw,
        "features": None, "prediction": None, "prediction_label": None,
        "probability": None, "decision_threshold": settings.decision_threshold,
        "model_name": settings.model_name, "model_alias": settings.model_alias,
        "model_version": current.version if current else None,
        "model_run_id": current.run_id if current else None,
        "client_host": request.client.host if request.client else None,
    }

    if current is None:
        PREDICTION_ERRORS.labels("model_not_loaded").inc()
        background_tasks.add_task(db.log_inference, {
            **base_log, "status": "error", "error_message": "model_not_loaded",
            "latency_ms": (time.perf_counter() - start) * 1000})
        raise HTTPException(status_code=503, detail=(
            f"No hay modelo productivo (@{settings.model_alias}) en MLflow todavía. "
            "Ejecute el DAG de Airflow."))

    try:
        features = build_features(pd.DataFrame([raw]))
        with MODEL_LATENCY.time():
            probability = float(current.model.predict_proba(features)[0, 1])
    except Exception as exc:  # noqa: BLE001
        PREDICTION_ERRORS.labels("inference_error").inc()
        log.exception("Error de inferencia")
        background_tasks.add_task(db.log_inference, {
            **base_log, "status": "error", "error_message": str(exc)[:500],
            "latency_ms": (time.perf_counter() - start) * 1000})
        raise HTTPException(status_code=500, detail="Error al ejecutar el modelo") from exc

    prediction = int(probability >= settings.decision_threshold)
    label = LABELS[prediction]
    elapsed_ms = (time.perf_counter() - start) * 1000
    PREDICTIONS.labels(current.name, current.version, label).inc()
    PREDICTED_PROBABILITY.observe(probability)

    background_tasks.add_task(db.log_inference, {
        **base_log, "features": features.iloc[0].to_dict(), "prediction": prediction,
        "prediction_label": label, "probability": probability,
        "latency_ms": elapsed_ms, "status": "success", "error_message": None})

    return PredictionResponse(
        request_id=request_id, prediction=prediction, prediction_label=label,
        probability=round(probability, 6), decision_threshold=settings.decision_threshold,
        model_name=current.name, model_version=current.version,
        model_alias=current.alias, processing_time_ms=round(elapsed_ms, 3))


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
