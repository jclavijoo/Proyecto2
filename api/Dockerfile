#   cd api
#   docker build -t <usuario>/mlops-p2-api:0.1.0 .
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py /app/

EXPOSE 8000

# 1 worker por pod: se escala con réplicas en Kubernetes
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
