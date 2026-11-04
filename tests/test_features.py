"""Pruebas unitarias de la ingeniería de características (pytest -q tests/)."""
import hashlib
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FEATURES = ROOT / "airflow" / "dags" / "diabetes_pipeline" / "features.py"
API_FEATURES = ROOT / "api" / "features.py"

spec = importlib.util.spec_from_file_location("features", API_FEATURES)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


def test_entrenamiento_e_inferencia_usan_el_mismo_codigo():
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    assert digest(TRAIN_FEATURES) == digest(API_FEATURES), (
        "airflow/dags/diabetes_pipeline/features.py y api/features.py deben ser idénticos")


def _row(**overrides):
    row = {
        "encounter_id": "1", "patient_nbr": "10", "race": "Caucasian",
        "gender": "Female", "age": "[70-80)", "admission_type_id": "1",
        "discharge_disposition_id": "1", "admission_source_id": "7",
        "time_in_hospital": "3", "medical_specialty": "?",
        "num_lab_procedures": "41", "num_procedures": "0",
        "num_medications": "11", "number_outpatient": "0",
        "number_emergency": "1", "number_inpatient": "2", "diag_1": "428",
        "diag_2": "250.01", "diag_3": "V45", "number_diagnoses": "9",
        "max_glu_serum": "None", "A1Cresult": ">8", "change": "Ch",
        "diabetesMed": "Yes", "readmitted": "<30",
    }
    row.update({m: "No" for m in features.MEDICATION_COLUMNS})
    row.update({"insulin": "Up", "metformin": "Steady"})
    row.update(overrides)
    return row


def test_build_features():
    X = features.build_features(pd.DataFrame([_row()]))
    assert list(X.columns) == features.FEATURE_COLUMNS
    assert X.isna().sum().sum() == 0
    r = X.iloc[0]
    assert r.age_years == 75
    assert r.diag_1_group == "Circulatory" and r.diag_2_group == "Diabetes"
    assert r.service_utilization == 3
    assert r.num_meds_changed == 1 and r.num_meds_active == 2


def test_objetivo_y_exclusiones():
    df = pd.DataFrame([_row(), _row(readmitted="NO", discharge_disposition_id="11"),
                       _row(gender="Unknown/Invalid")])
    assert features.build_target(df).tolist() == [1, 0, 1]
    assert features.exclusion_reason(df).tolist() == [None, "expired_or_hospice", "invalid_gender"]


def test_particion_deterministica():
    assert features.assign_split(123) == features.assign_split("123")
    share = pd.Series([features.assign_split(i) for i in range(5000)]).value_counts(normalize=True)
    assert 0.65 < share["train"] < 0.75

def test_():
    pytest.raises 
    pass

def test_2():
    pass
