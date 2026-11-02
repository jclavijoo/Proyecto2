"""initContainer de MLflow: espera a MinIO y crea el bucket de artefactos si no existe."""
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

endpoint = os.environ["MLFLOW_S3_ENDPOINT_URL"]
bucket = os.environ.get("MLFLOW_ARTIFACTS_BUCKET", "mlflow-artifacts")
s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)

for attempt in range(60):
    try:
        existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
        if bucket not in existing:
            s3.create_bucket(Bucket=bucket)
            print(f"Bucket '{bucket}' creado en {endpoint}")
        else:
            print(f"Bucket '{bucket}' ya existe")
        sys.exit(0)
    except (EndpointConnectionError, ClientError) as exc:
        print(f"[{attempt}] MinIO no disponible todavía: {exc}")
        time.sleep(5)

print("No fue posible conectarse a MinIO")
sys.exit(1)
