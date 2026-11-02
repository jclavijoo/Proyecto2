"""Configuración del pipeline leída de variables de entorno.

Los valores llegan desde el ConfigMap `pipeline-config` y el Secret
`postgres-data-secret` (ver airflow/values/airflow-values.yaml).
"""
import os

# Fuente de datos
SOURCE_FILE_PATH = os.getenv("SOURCE_FILE_PATH", "/opt/airflow/data/Diabetes.csv")
SOURCE_FILE_URL = os.getenv("SOURCE_FILE_URL", "")
MAX_BATCH_SIZE = 15_000  # límite impuesto por el enunciado
BATCH_SIZE = min(int(os.getenv("BATCH_SIZE", "15000")), MAX_BATCH_SIZE)

# Base de datos de datos (RAW / CLEAN / INFERENCE)
DATA_DB = {
    "host": os.getenv("DATA_DB_HOST", "postgres-data"),
    "port": int(os.getenv("DATA_DB_PORT", "5432")),
    "dbname": os.getenv("DATA_DB_NAME", "diabetes"),
    "user": os.getenv("DATA_DB_USER", "mlops"),
    "password": os.getenv("DATA_DB_PASSWORD", ""),
}

# Calidad de datos
MAX_REJECT_RATIO = float(os.getenv("MAX_REJECT_RATIO", "0.10"))

# Particiones
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.70"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.15"))
MIN_POSITIVES_PER_SPLIT = int(os.getenv("MIN_POSITIVES_PER_SPLIT", "20"))

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "diabetes-readmission")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-readmission")
PRODUCTION_ALIAS = os.getenv("PRODUCTION_ALIAS", "champion")
CHALLENGER_ALIAS = os.getenv("CHALLENGER_ALIAS", "challenger")
PREVIOUS_ALIAS = os.getenv("PREVIOUS_ALIAS", "previous-champion")

# Selección de modelos
# Métrica principal (sobre la clase positiva "<30"): f1 | recall | precision | roc_auc | pr_auc
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "f1")
MIN_IMPROVEMENT = float(os.getenv("MIN_IMPROVEMENT", "0.0"))
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
CANDIDATE_MODELS = [
    m.strip() for m in os.getenv(
        "CANDIDATE_MODELS", "logistic_regression,random_forest,hist_gradient_boosting"
    ).split(",") if m.strip()
]
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
