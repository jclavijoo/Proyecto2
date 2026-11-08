"""Configuración común de las pruebas.

Las variables se definen antes de importar la API: así no intenta registrar
inferencias en PostgreSQL ni conectarse a un MLflow real.
"""
import os

os.environ.setdefault("LOG_INFERENCES", "false")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://mlflow-no-disponible:5000")
