"""Registro de modelos, comparación contra el campeón y promoción por alias.

Flujo:
  1. register_best_candidate: elige el mejor candidato de la ejecución por la
     métrica principal en VALIDACIÓN y lo registra como nueva versión con el
     alias `challenger`.
  2. compare_with_champion: re-evalúa el campeón actual (alias `champion`) sobre
     el conjunto de PRUEBA vigente (que crece con cada lote) y lo compara con el
     challenger sobre exactamente los mismos datos -> comparación justa.
  3. promote / keep: mueve el alias `champion` (y guarda el anterior como
     `previous-champion` para rollback) o deja el campeón actual.

La API siempre carga `models:/<MODEL_NAME>@champion`, por lo que no hay que
cambiar código ni copiar archivos para cambiar de modelo.
"""
from __future__ import annotations

import logging
import time

from diabetes_pipeline import config
from diabetes_pipeline.training import compute_metrics, load_splits

log = logging.getLogger(__name__)


def _client():
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)


def _wait_until_ready(client, name: str, version: str, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        mv = client.get_model_version(name, version)
        if mv.status == "READY":
            return
        time.sleep(2)
    raise TimeoutError(f"La versión {version} de {name} no quedó READY")


def register_best_candidate(candidates: list[dict], dag_run_id: str) -> dict:
    import mlflow

    if not candidates:
        raise ValueError("No hay candidatos entrenados")
    client = _client()
    best = max(candidates, key=lambda c: c["val_score"])
    log.info("Mejor candidato (val_%s=%.4f): %s", config.PRIMARY_METRIC, best["val_score"], best)

    # En MLflow 3 el modelo es un LoggedModel (models:/m-...) vinculado al run
    mv = mlflow.register_model(best["model_uri"], config.MODEL_NAME)
    _wait_until_ready(client, config.MODEL_NAME, mv.version)

    client.update_model_version(
        config.MODEL_NAME, mv.version,
        description=(f"{best['model_key']} entrenado por Airflow ({dag_run_id}). "
                     f"val_{config.PRIMARY_METRIC}={best['val_score']:.4f}"))
    for key, value in {
        "model_key": best["model_key"], "airflow_dag_run_id": dag_run_id,
        "primary_metric": config.PRIMARY_METRIC,
        f"val_{config.PRIMARY_METRIC}": f"{best['val_score']:.4f}",
        f"test_{config.PRIMARY_METRIC}": f"{best['test_score']:.4f}",
        "decision_threshold": str(config.DECISION_THRESHOLD),
        "candidates_compared": ",".join(c["model_key"] for c in candidates),
    }.items():
        client.set_model_version_tag(config.MODEL_NAME, mv.version, key, value)
    client.set_registered_model_alias(config.MODEL_NAME, config.CHALLENGER_ALIAS, mv.version)

    return {**best, "version": str(mv.version)}


def compare_with_champion(candidate: dict) -> dict:
    import mlflow.sklearn
    from mlflow.exceptions import MlflowException

    client = _client()
    metric = config.PRIMARY_METRIC
    result = {"candidate": candidate, "metric": metric,
              "candidate_test_score": candidate["test_score"],
              "champion_version": None, "champion_test_score": None}

    try:
        champion = client.get_model_version_by_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS)
    except MlflowException:
        champion = None

    if champion is None:
        result.update(promote=True, reason="No existe modelo productivo: se promueve el primero.")
    elif str(champion.version) == candidate["version"]:
        result.update(promote=False, reason="El candidato ya es el campeón.")
    else:
        X_test, y_test = load_splits()["test"]
        model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/{champion.version}")
        champ_score = compute_metrics(y_test, model.predict_proba(X_test)[:, 1])[metric]
        better = candidate["test_score"] > champ_score + config.MIN_IMPROVEMENT
        result.update(
            champion_version=str(champion.version),
            champion_test_score=float(champ_score),
            promote=bool(better),
            reason=(f"challenger test_{metric}={candidate['test_score']:.4f} vs "
                    f"champion v{champion.version} test_{metric}={champ_score:.4f} "
                    f"(mejora mínima {config.MIN_IMPROVEMENT})"),
        )
        # Trazabilidad: la comparación queda en el run del candidato
        client.log_metric(candidate["run_id"], f"champion_test_{metric}", champ_score)

    client.set_model_version_tag(config.MODEL_NAME, candidate["version"],
                                 "comparison", result["reason"])
    log.info("Comparación: %s", result)
    return result


def promote(comparison: dict) -> dict:
    client = _client()
    version = comparison["candidate"]["version"]
    if comparison.get("champion_version"):
        client.set_registered_model_alias(
            config.MODEL_NAME, config.PREVIOUS_ALIAS, comparison["champion_version"])
        client.set_model_version_tag(config.MODEL_NAME, comparison["champion_version"],
                                     "status", "archived")
    client.set_registered_model_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS, version)
    client.set_model_version_tag(config.MODEL_NAME, version, "status", "production")
    client.set_model_version_tag(config.MODEL_NAME, version, "promoted_at",
                                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    log.info("Versión %s promovida a @%s", version, config.PRODUCTION_ALIAS)
    return {"promoted_version": version, "alias": config.PRODUCTION_ALIAS}


def keep_champion(comparison: dict) -> dict:
    client = _client()
    client.set_model_version_tag(config.MODEL_NAME, comparison["candidate"]["version"],
                                 "status", "rejected")
    log.info("Se mantiene el campeón v%s", comparison.get("champion_version"))
    return {"kept_version": comparison.get("champion_version")}
