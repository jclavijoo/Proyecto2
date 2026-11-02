"""Carga controlada del modelo productivo desde el Model Registry de MLflow.

Estrategia (documentar en la sustentación):
  * Al iniciar, la API resuelve `models:/<MODEL_NAME>@<MODEL_ALIAS>` a una
    versión concreta y la carga UNA vez en memoria.
  * Un hilo en segundo plano consulta cada MODEL_REFRESH_SECONDS a qué versión
    apunta el alias (una llamada liviana a la API REST de MLflow). Solo si
    cambió, descarga la nueva versión y la intercambia de forma atómica
    (hot-swap): las solicitudes en curso terminan con el modelo anterior.
  * POST /reload-model fuerza la verificación inmediata.
  * Si MLflow cae, la API sigue sirviendo con el último modelo cargado.
Nunca se descarga el modelo por solicitud y no hay rutas ni versiones quemadas.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import settings
from metrics import MODEL_INFO, MODEL_LOADED, MODEL_RELOADS

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    name: str
    alias: str
    version: str
    run_id: str
    loaded_at: str
    tags: dict = field(default_factory=dict)


class ModelManager:
    def __init__(self) -> None:
        self._current: Optional[LoadedModel] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.last_check_at: Optional[str] = None
        self.last_error: Optional[str] = None

    @property
    def current(self) -> Optional[LoadedModel]:
        return self._current

    def refresh(self, force: bool = False) -> bool:
        """Devuelve True si se cargó una versión nueva."""
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient

        with self._lock:
            self.last_check_at = _now()
            try:
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                client = MlflowClient()
                mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
                version = str(mv.version)
                if not force and self._current and self._current.version == version:
                    self.last_error = None
                    return False

                log.info("Cargando %s v%s (@%s)", settings.model_name, version, settings.model_alias)
                model = mlflow.sklearn.load_model(f"models:/{settings.model_name}/{version}")
                if not hasattr(model, "predict_proba"):
                    raise TypeError("El modelo registrado no expone predict_proba")

                self._current = LoadedModel(
                    model=model, name=settings.model_name, alias=settings.model_alias,
                    version=version, run_id=mv.run_id, loaded_at=_now(), tags=dict(mv.tags or {}))
                MODEL_INFO.clear()
                MODEL_INFO.labels(settings.model_name, version, settings.model_alias, mv.run_id).set(1)
                MODEL_LOADED.set(1)
                MODEL_RELOADS.labels("success").inc()
                self.last_error = None
                return True
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                MODEL_RELOADS.labels("error").inc()
                MODEL_LOADED.set(1 if self._current else 0)
                log.warning("No se pudo actualizar el modelo: %s", self.last_error)
                return False

    def start_background_refresh(self) -> None:
        def loop() -> None:
            while not self._stop.wait(settings.model_refresh_seconds):
                self.refresh()

        threading.Thread(target=loop, name="model-refresh", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


manager = ModelManager()
