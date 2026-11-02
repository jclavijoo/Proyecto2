# Basado en Niveles/3/locust/Dockerfile.locust
#   cd locust
#   docker build -t <usuario>/mlops-p2-locust:0.1.0 .
FROM python:3.12-slim

WORKDIR /locust

COPY requirements.txt /locust/
RUN pip install --no-cache-dir -r requirements.txt

COPY locustfile.py /locust/

EXPOSE 8089

CMD ["locust", "-f", "locustfile.py"]
