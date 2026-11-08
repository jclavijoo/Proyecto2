"""Transformaciones compartidas entre entrenamiento (Airflow) e inferencia (API).

IMPORTANTE: este archivo existe en DOS lugares y debe ser idéntico:
  * airflow/dags/diabetes_pipeline/features.py  (entrenamiento)
  * api/features.py                             (inferencia)
Así el modelo recibe exactamente las mismas columnas en ambos lados (evita el
"training/serving skew"). tests/test_features_identicos.py verifica que sean iguales.

Entrada: un DataFrame con las columnas originales del CSV (como texto o tipadas).
Salida: un DataFrame con FEATURE_COLUMNS listo para el Pipeline de scikit-learn.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np
import pandas as pd

TRANSFORM_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Columnas del archivo fuente
# --------------------------------------------------------------------------- #
ID_COLUMNS = ["encounter_id", "patient_nbr"]
TARGET_RAW_COLUMN = "readmitted"
TARGET_COLUMN = "readmitted_30"
POSITIVE_RAW_VALUE = "<30"
VALID_TARGET_VALUES = {"NO", ">30", "<30"}

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]
MEDICATION_VALUES = {"No", "Steady", "Up", "Down"}

# Columnas que la API recibe (formato "crudo", igual que el CSV)
RAW_INPUT_COLUMNS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses", "max_glu_serum",
    "A1Cresult", *MEDICATION_COLUMNS, "change", "diabetesMed",
]

# Columnas mínimas que debe traer el archivo fuente
REQUIRED_SOURCE_COLUMNS = ID_COLUMNS + RAW_INPUT_COLUMNS + [TARGET_RAW_COLUMN]

# Columnas descartadas y justificación (documentar también en el README)
DROPPED_COLUMNS = {
    "weight": "~97% de valores faltantes ('?'): no aporta información fiable.",
    "payer_code": "~40% faltante y es administrativa (aseguradora), no clínica.",
    "examide": "Constante ('No') en todo el dataset: varianza cero.",
    "citoglipton": "Constante ('No') en todo el dataset: varianza cero.",
    "encounter_id": "Identificador: se usa como llave, no como variable.",
    "patient_nbr": "Identificador: solo se usa para la partición agrupada.",
}

# --------------------------------------------------------------------------- #
# Columnas de salida (deben coincidir con db/02_clean.sql)
# --------------------------------------------------------------------------- #
CATEGORICAL_FEATURES = [
    "race", "gender", "admission_type", "discharge_group",
    "admission_source_group", "medical_specialty_group",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "max_glu_serum", "a1c_result", "insulin", "metformin",
]
NUMERIC_FEATURES = [
    "age_years", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "service_utilization",
    "num_meds_changed", "num_meds_active", "med_change", "diabetes_med",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# --------------------------------------------------------------------------- #
# Diccionarios de agrupación
# --------------------------------------------------------------------------- #
# IDs de alta que corresponden a fallecimiento u hospicio: el paciente no puede
# ser readmitido, por lo que se excluyen del entrenamiento.
EXCLUDED_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

ADMISSION_TYPE_MAP = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Other",
                      7: "Other", 5: "Unknown", 6: "Unknown", 8: "Unknown"}

TOP_SPECIALTIES = {
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
    "Cardiology", "Nephrology", "Orthopedics", "Orthopedics-Reconstructive",
    "Radiologist", "Pulmonology", "Psychiatry", "Urology",
    "ObstetricsandGynecology", "Gastroenterology",
}

MISSING_TOKENS = {"", "?", "nan", "NaN", "NULL", "null", "None"}


def _to_text(value: object) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _as_str(series: pd.Series) -> pd.Series:
    # Conversión elemento a elemento: mucho más rápida que el dtype "string" de
    # pandas cuando la API transforma una sola fila (perfilado: ~5x).
    values = [_to_text(v) for v in series.to_numpy(dtype=object)]
    return pd.Series(values, index=series.index, dtype=object)


def _as_int(series: pd.Series, default: int = 0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(int)


def clean_text(value: object) -> str:
    """Limpieza de texto: quita espacios y convierte los marcadores de faltante
    del dataset ('?', 'NULL', 'None', NaN, vacío) en una cadena vacía."""
    text = _to_text(value)
    return "" if text in MISSING_TOKENS else text


def _clean_category(series: pd.Series, missing: str = "Unknown") -> pd.Series:
    values = [clean_text(v) or missing for v in series.to_numpy(dtype=object)]
    return pd.Series(values, index=series.index, dtype=object).astype(str)


def _lab_result(series: pd.Series) -> pd.Series:
    """max_glu_serum / A1Cresult: 'None' significa 'no medido' (categoría válida)."""
    s = _as_str(series)
    return s.where(~s.isin(MISSING_TOKENS), "None").astype(str)


_AGE_PATTERN = re.compile(r"^\[(\d+)-(\d+)\)$")


def parse_age_bracket(text: object) -> float:
    """Convierte un rango de edad del dataset en su punto medio: '[70-80)' -> 75.0.

    Lanza ValueError si el texto no tiene el formato '[a-b)' o si a >= b.
    """
    match = _AGE_PATTERN.match(clean_text(text))
    if not match:
        raise ValueError(f"Rango de edad inválido: {text!r}")
    low, high = int(match.group(1)), int(match.group(2))
    if low >= high:
        raise ValueError(f"Rango de edad inválido (inicio >= fin): {text!r}")
    return (low + high) / 2.0


def _age_to_years(series: pd.Series) -> pd.Series:
    def parse(value: object) -> float:
        try:
            return parse_age_bracket(value)
        except ValueError:
            return np.nan

    years = series.map(parse).astype(float)
    return years.fillna(years.median() if years.notna().any() else 55.0)


def icd9_group(code: object) -> str:
    """Agrupa un código ICD-9 en una categoría clínica (Strack et al., 2014)."""
    s = str(code).strip() if code is not None else ""
    if s in MISSING_TOKENS:
        return "Missing"
    if s[0] in ("V", "E"):
        return "Other"
    try:
        value = float(s)
    except ValueError:
        return "Other"
    integer = int(value)
    if 250 <= value < 251:
        return "Diabetes"
    if 390 <= value <= 459 or integer == 785:
        return "Circulatory"
    if 460 <= value <= 519 or integer == 786:
        return "Respiratory"
    if 520 <= value <= 579 or integer == 787:
        return "Digestive"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or integer == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def _discharge_group(ids: pd.Series) -> pd.Series:
    def group(v: int) -> str:
        if v == 1:
            return "Home"
        if v in (6, 8):
            return "HomeHealth"
        if v in (2, 3, 4, 5, 22, 23, 24, 27, 28, 29, 30):
            return "Facility"
        if v == 7:
            return "LeftAMA"
        if v in (18, 25, 26):
            return "Unknown"
        return "Other"
    return ids.map(group)


def _admission_source_group(ids: pd.Series) -> pd.Series:
    def group(v: int) -> str:
        if v == 7:
            return "Emergency"
        if v in (1, 2, 3):
            return "Referral"
        if v in (4, 5, 6, 10, 22, 25):
            return "Transfer"
        if v in (9, 15, 17, 20, 21):
            return "Unknown"
        return "Other"
    return ids.map(group)


def _specialty_group(series: pd.Series) -> pd.Series:
    s = _clean_category(series, missing="Unknown")

    def group(v: str) -> str:
        if v == "Unknown" or v in TOP_SPECIALTIES:
            return v
        if v.startswith("Surgery"):
            return "Surgery"
        return "Other"
    return s.map(group)


def _medication_matrix(df: pd.DataFrame) -> np.ndarray:
    """Matriz (n_filas x 23) con los valores de medicación normalizados (numpy)."""
    columns = [
        [_to_text(v) for v in df[col].to_numpy(dtype=object)] if col in df.columns
        else ["No"] * len(df)
        for col in MEDICATION_COLUMNS
    ]
    matrix = np.array(columns, dtype=object).reshape(len(MEDICATION_COLUMNS), len(df)).T
    return np.where(np.isin(matrix, list(MEDICATION_VALUES)), matrix, "No")


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #
# Rangos válidos (inclusive). Son los mismos que valida la API en schemas.py;
# tests/test_api.py comprueba que no se desalineen.
NUMERIC_RANGES = {
    "admission_type_id": (1, 8),
    "discharge_disposition_id": (1, 30),
    "admission_source_id": (1, 26),
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (0, 200),
    "num_procedures": (0, 10),
    "num_medications": (0, 100),
    "number_outpatient": (0, 100),
    "number_emergency": (0, 100),
    "number_inpatient": (0, 100),
    "number_diagnoses": (1, 20),
}


def numeric_range_violations(df: pd.DataFrame) -> pd.Series:
    """Validación de rangos numéricos por fila.

    Devuelve, para cada fila, el motivo del primer valor inválido
    ('<columna>_fuera_de_rango') o None si todos los valores son válidos.
    Un valor es inválido si no es numérico, no es entero o está fuera de
    NUMERIC_RANGES. Lanza ValueError si falta alguna de las columnas.
    """
    missing = [c for c in NUMERIC_RANGES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas numéricas: {missing}")
    reason = pd.Series([None] * len(df), index=df.index, dtype=object)
    for col, (low, high) in NUMERIC_RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.isna() | (values < low) | (values > high) | (values != values.round())
        reason[bad & reason.isna()] = f"{col}_fuera_de_rango"
    return reason


def missing_required_columns(columns: Iterable[str]) -> list[str]:
    present = set(columns)
    return [c for c in REQUIRED_SOURCE_COLUMNS if c not in present]


def exclusion_reason(df: pd.DataFrame) -> pd.Series:
    """Motivo de exclusión del entrenamiento por fila (None si la fila se conserva)."""
    reason = pd.Series([None] * len(df), index=df.index, dtype=object)
    discharge = _as_int(df["discharge_disposition_id"], default=-1)
    reason[discharge.isin(EXCLUDED_DISCHARGE_IDS)] = "expired_or_hospice"
    gender = _as_str(df["gender"])
    reason[~gender.isin({"Female", "Male"}) & reason.isna()] = "invalid_gender"
    return reason


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas crudas en FEATURE_COLUMNS (sin nulos)."""
    out = pd.DataFrame(index=df.index)
    meds = _medication_matrix(df)

    out["race"] = _clean_category(df["race"])
    out["gender"] = _clean_category(df["gender"])
    out["admission_type"] = (
        _as_int(df["admission_type_id"], -1).map(ADMISSION_TYPE_MAP).fillna("Unknown")
    )
    out["discharge_group"] = _discharge_group(_as_int(df["discharge_disposition_id"], -1))
    out["admission_source_group"] = _admission_source_group(_as_int(df["admission_source_id"], -1))
    out["medical_specialty_group"] = _specialty_group(df["medical_specialty"])
    for i in (1, 2, 3):
        out[f"diag_{i}_group"] = df[f"diag_{i}"].map(icd9_group)
    out["max_glu_serum"] = _lab_result(df["max_glu_serum"])
    out["a1c_result"] = _lab_result(df["A1Cresult"])
    out["insulin"] = meds[:, MEDICATION_COLUMNS.index("insulin")].astype(str)
    out["metformin"] = meds[:, MEDICATION_COLUMNS.index("metformin")].astype(str)

    out["age_years"] = _age_to_years(df["age"])
    for col in ("time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency",
                "number_inpatient", "number_diagnoses"):
        out[col] = _as_int(df[col])
    out["service_utilization"] = (
        out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    )
    out["num_meds_changed"] = np.isin(meds, ["Up", "Down"]).sum(axis=1).astype(int)
    out["num_meds_active"] = (meds != "No").sum(axis=1).astype(int)
    out["med_change"] = (_as_str(df["change"]) == "Ch").astype(int)
    out["diabetes_med"] = (_as_str(df["diabetesMed"]) == "Yes").astype(int)

    return out[FEATURE_COLUMNS]


def build_target(df: pd.DataFrame) -> pd.Series:
    """1 si el paciente fue readmitido en menos de 30 días, 0 en otro caso."""
    return (_as_str(df[TARGET_RAW_COLUMN]) == POSITIVE_RAW_VALUE).astype(int)


def assign_split(patient_nbr: object, train_ratio: float = 0.70,
                 val_ratio: float = 0.15) -> str:
    """Partición determinística AGRUPADA por paciente.

    Un mismo paciente siempre cae en la misma partición aunque llegue en lotes
    distintos -> no hay fuga de información entre train/val/test y la partición
    es estable entre ejecuciones del DAG (reproducible).
    """
    bucket = int(hashlib.md5(str(patient_nbr).encode()).hexdigest(), 16) % 10_000 / 10_000
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"
