"""Prueba de carga de la API de inferencia.

Escenario: usuarios clínicos que envían encuentros aleatorios (pero válidos) a
/predict y ocasionalmente consultan /model-info y /health.

Modos:
  * UI (por defecto): defina usuarios y spawn rate en http://localhost:8089.
  * Escalonado: LOCUST_STEP_LOAD=true sube la carga en escalones para encontrar
    el punto aproximado de degradación (p95 creciente / errores).
      LOCUST_STEP_USERS=10  usuarios que se agregan por escalón
      LOCUST_STEP_SECONDS=60 duración de cada escalón
      LOCUST_MAX_USERS=200  tope de usuarios
"""
import os
import random

from locust import HttpUser, LoadTestShape, between, task

AGES = ["[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]
RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"]
DIAGS = ["428", "250.02", "414", "786", "486", "410", "427", "599", "276", "401", "V45", "715"]
SPECIALTIES = ["InternalMedicine", "Cardiology", "Emergency/Trauma", "Family/GeneralPractice",
               "Surgery-General", "Orthopedics", "?"]
MEDS = ["No", "No", "No", "Steady", "Up", "Down"]


def random_patient() -> dict:
    return {
        "race": random.choice(RACES),
        "gender": random.choice(["Female", "Male"]),
        "age": random.choice(AGES),
        "admission_type_id": random.choice([1, 1, 2, 3, 5, 6]),
        "discharge_disposition_id": random.choice([1, 1, 1, 3, 6, 2, 22]),
        "admission_source_id": random.choice([7, 7, 1, 4, 6]),
        "time_in_hospital": random.randint(1, 14),
        "medical_specialty": random.choice(SPECIALTIES),
        "num_lab_procedures": random.randint(1, 100),
        "num_procedures": random.randint(0, 6),
        "num_medications": random.randint(1, 60),
        "number_outpatient": random.choice([0, 0, 0, 1, 2]),
        "number_emergency": random.choice([0, 0, 0, 1, 3]),
        "number_inpatient": random.choice([0, 0, 1, 2, 4]),
        "diag_1": random.choice(DIAGS),
        "diag_2": random.choice(DIAGS),
        "diag_3": random.choice(DIAGS + ["?"]),
        "number_diagnoses": random.randint(1, 16),
        "max_glu_serum": random.choice(["None", "None", "Norm", ">200", ">300"]),
        "A1Cresult": random.choice(["None", "None", "Norm", ">7", ">8"]),
        "metformin": random.choice(MEDS),
        "insulin": random.choice(MEDS),
        "glipizide": random.choice(MEDS),
        "change": random.choice(["No", "Ch"]),
        "diabetesMed": random.choice(["No", "Yes", "Yes"]),
    }


class ClinicalUser(HttpUser):
    wait_time = between(float(os.getenv("LOCUST_WAIT_MIN", "0.5")),
                        float(os.getenv("LOCUST_WAIT_MAX", "1.5")))

    @task(10)
    def predict(self):
        with self.client.post("/predict", json=random_patient(), name="/predict",
                              catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:150]}")
                return
            body = resp.json()
            if "prediction" not in body or "model_version" not in body:
                resp.failure("Respuesta sin 'prediction' o 'model_version'")
            else:
                resp.success()

    @task(1)
    def model_info(self):
        self.client.get("/model-info", name="/model-info")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")


if os.getenv("LOCUST_STEP_LOAD", "false").lower() == "true":

    class StepLoadShape(LoadTestShape):
        step_users = int(os.getenv("LOCUST_STEP_USERS", "10"))
        step_seconds = int(os.getenv("LOCUST_STEP_SECONDS", "60"))
        max_users = int(os.getenv("LOCUST_MAX_USERS", "200"))

        def tick(self):
            run_time = self.get_run_time()
            users = (int(run_time // self.step_seconds) + 1) * self.step_users
            if users > self.max_users:
                return None  # fin de la prueba
            return users, self.step_users
