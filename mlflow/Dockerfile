# Basado en Niveles/2/mlflow_compose/mlflow/Dockerfile, con PostgreSQL en vez de SQLite
#   cd mlflow
#   docker build -t <usuario>/mlops-p2-mlflow:0.1.0 .
FROM python:3.12-slim

# sqlalchemy fijo: MLflow 3.4 falla con SQLAlchemy 2.1
RUN pip install --no-cache-dir mlflow==3.4.0 sqlalchemy==2.0.36 psycopg2-binary==2.9.10 boto3==1.35.99

WORKDIR /app
COPY ensure_bucket.py /app/

EXPOSE 5000

# El comando completo (PostgreSQL + MinIO) está en k8s/mlflow-deployment.yaml y en docker-compose.yaml
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
