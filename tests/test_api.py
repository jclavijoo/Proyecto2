"""Pruebas de la API con un modelo de prueba (sin MLflow ni PostgreSQL)."""
import annotated_types
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import model_manager
from features import FEATURE_COLUMNS, NUMERIC_RANGES
from main import app
from schemas import PatientInput

EJEMPLO = PatientInput.model_config["json_schema_extra"]["example"]


class ModeloDePrueba:
    """Imita un Pipeline de scikit-learn: verifica columnas y devuelve 0.7."""

    def predict_proba(self, X):
        assert list(X.columns) == FEATURE_COLUMNS
        return np.array([[0.3, 0.7]])


@pytest.fixture
def cliente():
    # Sin "with": no se ejecuta el lifespan, así que la API no busca MLflow.
    return TestClient(app)


@pytest.fixture
def con_modelo(monkeypatch):
    cargado = model_manager.LoadedModel(
        model=ModeloDePrueba(), name="diabetes-readmission", alias="champion",
        version="7", run_id="run-de-prueba", loaded_at="2026-01-01T00:00:00+00:00")
    monkeypatch.setattr(model_manager.manager, "_current", cargado)


def test_schema_rechaza_valores_fuera_de_rango():
    with pytest.raises(ValidationError):
        PatientInput(**{**EJEMPLO, "time_in_hospital": 20})


def test_rangos_de_la_api_coinciden_con_los_del_pipeline():
    for columna, (minimo, maximo) in NUMERIC_RANGES.items():
        restricciones = PatientInput.model_fields[columna].metadata
        ge = next(r.ge for r in restricciones if isinstance(r, annotated_types.Ge))
        le = next(r.le for r in restricciones if isinstance(r, annotated_types.Le))
        assert (ge, le) == (minimo, maximo), columna


def test_health_responde_sin_modelo(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ok"


def test_predict_sin_modelo_responde_503(cliente, monkeypatch):
    monkeypatch.setattr(model_manager.manager, "_current", None)
    assert cliente.post("/predict", json=EJEMPLO).status_code == 503


def test_predict_con_modelo_devuelve_prediccion(cliente, con_modelo):
    respuesta = cliente.post("/predict", json=EJEMPLO)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["prediction"] == 1 and cuerpo["probability"] == 0.7
    assert cuerpo["model_version"] == "7" and cuerpo["model_alias"] == "champion"


def test_predict_con_datos_invalidos_responde_422(cliente, con_modelo):
    respuesta = cliente.post("/predict", json={**EJEMPLO, "gender": "X"})
    assert respuesta.status_code == 422
