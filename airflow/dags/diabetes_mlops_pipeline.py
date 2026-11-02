"""
### diabetes_mlops_pipeline

Pipeline incremental de MLOps (Proyecto 2 - Nivel 3).

En cada ejecución:
1. **check_source_file**: valida que el archivo fuente exista y tenga las columnas esperadas.
2. **load_raw_batch**: carga el siguiente lote (máx. 15.000 filas) a `raw.diabetes_raw`.
3. **has_data_to_process**: si no hay datos nuevos ni pendientes, se omite el resto (short-circuit).
4. **validate_raw_quality**: reglas de calidad -> filas VALID / REJECTED / DUPLICATE.
5. **preprocess_and_store_clean**: limpieza + ingeniería de características -> `clean.diabetes_clean`.
6. **split_datasets**: partición train/val/test determinística y agrupada por paciente.
7. **train_candidate** (mapeada): entrena N modelos en paralelo y registra cada run en MLflow.
8. **register_best_candidate**: registra el mejor como nueva versión (`@challenger`).
9. **compare_with_champion**: compara contra el modelo productivo sobre el mismo test.
10. **decide_promotion** -> **promote_candidate** (`@champion`) o **keep_current_champion**.

Periodicidad: `PIPELINE_SCHEDULE` (por defecto cada hora). `max_active_runs=1`
impide ejecuciones concurrentes que compitan por el mismo offset.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.sdk import Param, dag, get_current_context, task

SCHEDULE = os.getenv("PIPELINE_SCHEDULE", "@hourly") or None
CANDIDATE_MODELS = [m.strip() for m in os.getenv(
    "CANDIDATE_MODELS", "logistic_regression,random_forest,hist_gradient_boosting"
).split(",") if m.strip()]

default_args = {
    "owner": "mlops-equipo",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=30),
}


def _run_id() -> str:
    return get_current_context()["run_id"]


@dag(
    dag_id="diabetes_mlops_pipeline",
    description="Ingesta incremental, procesamiento, entrenamiento y promoción en MLflow",
    schedule=SCHEDULE,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    doc_md=__doc__,
    tags=["mlops", "proyecto2", "diabetes", "kubernetes"],
    params={
        "batch_size": Param(15000, type="integer", minimum=1, maximum=15000,
                            description="Filas por lote (máximo 15.000)"),
        "force_retrain": Param(False, type="boolean",
                               description="Reentrenar aunque no haya datos nuevos"),
    },
)
def diabetes_mlops_pipeline():

    @task
    def check_source_file() -> dict:
        from diabetes_pipeline.db import check_schema
        from diabetes_pipeline.ingestion import check_source_file as _check

        check_schema()
        return _check()

    @task
    def load_raw_batch(source: dict) -> dict:
        from diabetes_pipeline.ingestion import load_next_batch

        params = get_current_context()["params"]
        return load_next_batch(source, params["batch_size"], _run_id())

    @task.short_circuit
    def has_data_to_process(load_info: dict) -> bool:
        from diabetes_pipeline.ingestion import pending_rows

        force = get_current_context()["params"]["force_retrain"]
        pending = pending_rows()
        print(f"Filas nuevas: {load_info['rows_inserted']} | pendientes: {pending} | force: {force}")
        return bool(load_info["rows_inserted"] > 0 or pending > 0 or force)

    @task
    def validate_raw_quality() -> dict:
        from diabetes_pipeline.quality import validate_pending

        return validate_pending(_run_id())

    @task
    def preprocess_and_store_clean(_quality: dict) -> dict:
        from diabetes_pipeline.preprocessing import preprocess_valid_rows

        return preprocess_valid_rows()

    @task
    def split_datasets(_clean: dict) -> dict:
        from diabetes_pipeline.preprocessing import assign_splits

        info = assign_splits()
        info["total_rows"] = sum(s["rows"] for s in info["splits"].values())
        return info

    @task(max_active_tis_per_dagrun=2, execution_timeout=timedelta(minutes=45))
    def train_candidate(model_key: str, split_info: dict) -> dict:
        from diabetes_pipeline.training import train_candidate as _train

        return _train(model_key, _run_id(), split_info)

    @task
    def register_best_candidate(results: list[dict]) -> dict:
        from diabetes_pipeline.registry import register_best_candidate as _register

        return _register(list(results), _run_id())

    @task
    def compare_with_champion(candidate: dict) -> dict:
        from diabetes_pipeline.registry import compare_with_champion as _compare

        return _compare(candidate)

    @task.branch
    def decide_promotion(comparison: dict) -> str:
        print(comparison["reason"])
        return "promote_candidate" if comparison["promote"] else "keep_current_champion"

    @task
    def promote_candidate(comparison: dict) -> dict:
        from diabetes_pipeline.registry import promote

        return promote(comparison)

    @task
    def keep_current_champion(comparison: dict) -> dict:
        from diabetes_pipeline.registry import keep_champion

        return keep_champion(comparison)

    @task(trigger_rule="none_failed_min_one_success")
    def pipeline_summary() -> None:
        ti = get_current_context()["ti"]
        for task_id in ("load_raw_batch", "validate_raw_quality",
                        "preprocess_and_store_clean", "split_datasets",
                        "compare_with_champion"):
            print(task_id, "->", ti.xcom_pull(task_ids=task_id))

    # ------------------------------ dependencias ------------------------------ #
    source = check_source_file()
    loaded = load_raw_batch(source)
    gate = has_data_to_process(loaded)
    quality = validate_raw_quality()
    gate >> quality
    clean = preprocess_and_store_clean(quality)
    splits = split_datasets(clean)
    trained = train_candidate.partial(split_info=splits).expand(model_key=CANDIDATE_MODELS)
    candidate = register_best_candidate(trained)
    comparison = compare_with_champion(candidate)
    branch = decide_promotion(comparison)
    promoted = promote_candidate(comparison)
    kept = keep_current_champion(comparison)
    branch >> [promoted, kept] >> pipeline_summary()


diabetes_mlops_pipeline()
