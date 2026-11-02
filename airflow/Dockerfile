# Basado en Niveles/3/Airflow/proyecto_base/Dockerfile
#   cd airflow
#   docker build --build-arg AIRFLOW_BASE_TAG=3.1.8 -t <usuario>/mlops-p2-airflow:0.1.0 .
ARG AIRFLOW_BASE_TAG=3.1.8
FROM apache/airflow:${AIRFLOW_BASE_TAG}
ARG AIRFLOW_BASE_TAG

COPY requirements.txt /requirements.txt
# Se fija apache-airflow para que pip no cambie su versión al instalar las librerías
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_BASE_TAG}" -r /requirements.txt

COPY dags/ /opt/airflow/dags/
# data/ puede contener Diabetes.csv; si no está, el DAG lo descarga (SOURCE_FILE_URL)
COPY --chown=airflow:root data/ /opt/airflow/data/
