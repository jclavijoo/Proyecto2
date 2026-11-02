"""Entrenamiento de modelos candidatos y registro de experimentos en MLflow.

Cada candidato es un Pipeline de scikit-learn (preprocesamiento + modelo), de
modo que el artefacto registrado en MLflow recibe directamente FEATURE_COLUMNS.

Métrica principal (configurable con PRIMARY_METRIC, por defecto F1 de la clase
positiva "<30"): el problema está desbalanceado (~11% positivos), así que
accuracy es engañosa (un modelo que siempre dice "NO" logra ~89%). F1 equilibra
recall (detectar pacientes en riesgo) y precision (no saturar al equipo clínico
con falsas alarmas). ROC-AUC y PR-AUC se registran como métricas de apoyo
independientes del umbral.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from diabetes_pipeline import config
from diabetes_pipeline.db import read_df
from diabetes_pipeline.features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS,
                                   NUMERIC_FEATURES, TARGET_COLUMN, TRANSFORM_VERSION)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def load_splits() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    df = read_df(
        f"SELECT {', '.join(FEATURE_COLUMNS)}, {TARGET_COLUMN}, split "
        "FROM clean.diabetes_clean WHERE split IS NOT NULL")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col]).astype(float)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    out = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        out[name] = (part[FEATURE_COLUMNS].reset_index(drop=True),
                     part[TARGET_COLUMN].astype(int).reset_index(drop=True))
    return out


# --------------------------------------------------------------------------- #
# Modelos candidatos
# --------------------------------------------------------------------------- #
def _preprocessor(dense: bool, scale: bool) -> ColumnTransformer:
    onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                           sparse_output=not dense)
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("cat", onehot, CATEGORICAL_FEATURES), ("num", numeric, NUMERIC_FEATURES)],
        sparse_threshold=0.0 if dense else 0.3,
    )


def build_candidate(model_key: str) -> tuple[Pipeline, dict]:
    seed = config.RANDOM_SEED
    if model_key == "logistic_regression":
        params = {"C": 0.5, "class_weight": "balanced", "max_iter": 2000, "solver": "lbfgs"}
        return Pipeline([("prep", _preprocessor(dense=False, scale=True)),
                         ("model", LogisticRegression(random_state=seed, **params))]), params
    if model_key == "random_forest":
        params = {"n_estimators": 150, "max_depth": 12, "min_samples_leaf": 20,
                  "class_weight": "balanced_subsample", "n_jobs": 2}
        return Pipeline([("prep", _preprocessor(dense=False, scale=False)),
                         ("model", RandomForestClassifier(random_state=seed, **params))]), params
    if model_key == "hist_gradient_boosting":
        params = {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31,
                  "l2_regularization": 1.0, "class_weight": "balanced",
                  "early_stopping": True}
        return Pipeline([("prep", _preprocessor(dense=True, scale=False)),
                         ("model", HistGradientBoostingClassifier(random_state=seed, **params))]), params
    raise ValueError(f"Modelo candidato desconocido: {model_key}")


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: pd.Series, proba: np.ndarray,
                    threshold: float = config.DECISION_THRESHOLD) -> dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) > 1
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y_true, proba) if both_classes else float("nan"),
        "positive_rate_pred": float(y_pred.mean()),
    }


def _confusion_png(y_true, proba, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, (proba >= config.DECISION_THRESHOLD).astype(int))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], ["No <30", "<30"])
    ax.set_yticks([0, 1], ["No <30", "<30"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entrenamiento + MLflow
# --------------------------------------------------------------------------- #
def train_candidate(model_key: str, dag_run_id: str, data_info: dict) -> dict:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    splits = load_splits()
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    pipeline, params = build_candidate(model_key)
    log.info("Entrenando %s con %s filas", model_key, len(X_train))
    pipeline.fit(X_train, y_train)

    metrics: dict[str, float] = {}
    probas = {}
    for name, (X, y) in {"train": (X_train, y_train), "val": (X_val, y_val),
                         "test": (X_test, y_test)}.items():
        probas[name] = pipeline.predict_proba(X)[:, 1]
        metrics.update({f"{name}_{k}": float(v) for k, v in compute_metrics(y, probas[name]).items()})

    primary = config.PRIMARY_METRIC
    with mlflow.start_run(run_name=f"{model_key}__{dag_run_id}") as run:
        mlflow.set_tags({
            "airflow_dag_run_id": dag_run_id,
            "model_key": model_key,
            "primary_metric": primary,
            "transform_version": TRANSFORM_VERSION,
            "target": f"{TARGET_COLUMN} (readmitted == '<30')",
            "split_strategy": "hash(patient_nbr) agrupado 70/15/15",
        })
        mlflow.log_params({f"model__{k}": v for k, v in params.items()})
        mlflow.log_params({
            "model_key": model_key,
            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
            "train_positive_rate": round(float(y_train.mean()), 4),
            "n_features_in": len(FEATURE_COLUMNS),
            "decision_threshold": config.DECISION_THRESHOLD,
            "random_seed": config.RANDOM_SEED,
            "clean_rows_total": data_info.get("total_rows"),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_metric("selection_score", metrics[f"val_{primary}"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in ("val", "test"):
                _confusion_png(splits[name][1], probas[name], f"{model_key} - {name}",
                               tmp_path / f"confusion_matrix_{name}.png")
            (tmp_path / "classification_report_test.json").write_text(json.dumps(
                classification_report(
                    y_test, (probas["test"] >= config.DECISION_THRESHOLD).astype(int),
                    output_dict=True, zero_division=0), indent=2))
            (tmp_path / "data_splits.json").write_text(json.dumps(data_info, indent=2, default=str))
            (tmp_path / "features.json").write_text(json.dumps({
                "categorical": CATEGORICAL_FEATURES, "numeric": NUMERIC_FEATURES,
                "transform_version": TRANSFORM_VERSION}, indent=2))
            mlflow.log_artifacts(tmp, artifact_path="evaluation")

        signature = infer_signature(X_val.head(200), pipeline.predict_proba(X_val.head(200)))
        model_info = mlflow.sklearn.log_model(
            pipeline, name="model", signature=signature,
            input_example=X_val.head(3),
        )
        run_id = run.info.run_id

    result = {"model_key": model_key, "run_id": run_id, "model_uri": model_info.model_uri,
              "val_score": metrics[f"val_{primary}"],
              "test_score": metrics[f"test_{primary}"],
              "val_roc_auc": metrics["val_roc_auc"], "test_roc_auc": metrics["test_roc_auc"]}
    log.info("Candidato entrenado: %s", result)
    return result
