"""Esquemas de entrada/salida. La entrada usa el mismo formato del CSV original."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Med = Literal["No", "Steady", "Up", "Down"]
Age = Literal["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
              "[60-70)", "[70-80)", "[80-90)", "[90-100)"]


class PatientInput(BaseModel):
    """Encuentro hospitalario de un paciente diabético (columnas del dataset)."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={"example": {
            "race": "Caucasian", "gender": "Female", "age": "[70-80)",
            "admission_type_id": 1, "discharge_disposition_id": 1,
            "admission_source_id": 7, "time_in_hospital": 5,
            "medical_specialty": "InternalMedicine", "num_lab_procedures": 45,
            "num_procedures": 1, "num_medications": 18, "number_outpatient": 0,
            "number_emergency": 1, "number_inpatient": 2, "diag_1": "428",
            "diag_2": "250.02", "diag_3": "401", "number_diagnoses": 9,
            "max_glu_serum": "None", "A1Cresult": ">8", "metformin": "Steady",
            "insulin": "Up", "change": "Ch", "diabetesMed": "Yes",
        }},
    )

    race: Literal["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"] = "?"
    gender: Literal["Female", "Male"]
    age: Age
    admission_type_id: int = Field(ge=1, le=8)
    discharge_disposition_id: int = Field(ge=1, le=30)
    admission_source_id: int = Field(ge=1, le=26)
    time_in_hospital: int = Field(ge=1, le=14, description="Días de hospitalización")
    medical_specialty: str = Field("?", max_length=60)
    num_lab_procedures: int = Field(ge=0, le=200)
    num_procedures: int = Field(ge=0, le=10)
    num_medications: int = Field(ge=0, le=100)
    number_outpatient: int = Field(0, ge=0, le=100)
    number_emergency: int = Field(0, ge=0, le=100)
    number_inpatient: int = Field(0, ge=0, le=100)
    diag_1: str = Field(min_length=1, max_length=10, description="Código ICD-9")
    diag_2: str = Field("?", max_length=10)
    diag_3: str = Field("?", max_length=10)
    number_diagnoses: int = Field(ge=1, le=20)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = "None"
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = "None"
    metformin: Med = "No"
    repaglinide: Med = "No"
    nateglinide: Med = "No"
    chlorpropamide: Med = "No"
    glimepiride: Med = "No"
    acetohexamide: Med = "No"
    glipizide: Med = "No"
    glyburide: Med = "No"
    tolbutamide: Med = "No"
    pioglitazone: Med = "No"
    rosiglitazone: Med = "No"
    acarbose: Med = "No"
    miglitol: Med = "No"
    troglitazone: Med = "No"
    tolazamide: Med = "No"
    examide: Med = "No"
    citoglipton: Med = "No"
    insulin: Med = "No"
    glyburide_metformin: Med = Field("No", alias="glyburide-metformin")
    glipizide_metformin: Med = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: Med = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: Med = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: Med = Field("No", alias="metformin-pioglitazone")
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["No", "Yes"] = "No"


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    prediction: int = Field(description="1 = readmisión en <30 días")
    prediction_label: str
    probability: float = Field(description="Probabilidad de readmisión en <30 días")
    decision_threshold: float
    model_name: str
    model_version: str
    model_alias: str
    processing_time_ms: float


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["loaded", "not_loaded"]
    model_name: str
    model_alias: str
    model_version: Optional[str] = None
    run_id: Optional[str] = None
    loaded_at: Optional[str] = None
    last_check_at: Optional[str] = None
    last_error: Optional[str] = None
    refresh_interval_seconds: int
    version_tags: dict = {}
    mlflow_tracking_uri: str
