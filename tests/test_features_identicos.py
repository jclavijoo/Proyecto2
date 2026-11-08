"""El entrenamiento (Airflow) y la inferencia (API) deben usar el mismo preprocesamiento."""
import hashlib
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_features_es_identico_en_airflow_y_en_la_api():
    api = RAIZ / "api" / "features.py"
    airflow = RAIZ / "airflow" / "dags" / "diabetes_pipeline" / "features.py"
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    assert digest(api) == digest(airflow), (
        "airflow/dags/diabetes_pipeline/features.py y api/features.py deben ser idénticos")
