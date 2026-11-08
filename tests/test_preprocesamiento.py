"""Pruebas de las funciones de preprocesamiento (features.py):
limpieza de texto, validación de rangos numéricos y transformación de features."""
import math

import pandas as pd
import pytest

from features import (FEATURE_COLUMNS, MEDICATION_COLUMNS, NUMERIC_RANGES,
                      assign_split, build_features, build_target, clean_text,
                      exclusion_reason, icd9_group, numeric_range_violations,
                      parse_age_bracket)


def fila(**cambios):
    """Una fila con el formato del CSV original (todo como texto)."""
    base = {
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
    base.update({m: "No" for m in MEDICATION_COLUMNS})
    base.update({"insulin": "Up", "metformin": "Steady"})
    base.update(cambios)
    return base


# ----------------------------- limpieza de texto ----------------------------- #
def test_clean_text_quita_espacios():
    assert clean_text("  Caucasian  ") == "Caucasian"


@pytest.mark.parametrize("valor", ["?", "NULL", "None", "", "   ", None, float("nan")])
def test_clean_text_marcadores_de_faltante_quedan_vacios(valor):
    assert clean_text(valor) == ""


@pytest.mark.parametrize("texto, esperado", [("[0-10)", 5.0), ("[70-80)", 75.0), (" [90-100) ", 95.0)])
def test_parse_age_bracket_calcula_punto_medio(texto, esperado):
    assert parse_age_bracket(texto) == esperado


@pytest.mark.parametrize("texto", ["70-80", "[80-70)", "?", "", "setenta"])
def test_parse_age_bracket_formato_invalido_lanza_error(texto):
    with pytest.raises(ValueError):
        parse_age_bracket(texto)


# ------------------------- validación de rangos ------------------------------ #
def test_rangos_valores_en_los_limites_son_validos():
    filas = pd.DataFrame([fila(time_in_hospital="1"), fila(time_in_hospital="14")])
    assert numeric_range_violations(filas).isna().all()


@pytest.mark.parametrize("valor", ["0", "15", "2.5", "abc", ""])
def test_rangos_detecta_valores_invalidos(valor):
    motivo = numeric_range_violations(pd.DataFrame([fila(time_in_hospital=valor)]))
    assert motivo.iloc[0] == "time_in_hospital_fuera_de_rango"


def test_rangos_sin_columna_obligatoria_lanza_error():
    sin_columna = pd.DataFrame([fila()]).drop(columns=["num_medications"])
    with pytest.raises(ValueError, match="num_medications"):
        numeric_range_violations(sin_columna)


def test_rangos_cubren_todas_las_columnas_numericas_de_entrada():
    for columna, (minimo, maximo) in NUMERIC_RANGES.items():
        assert minimo < maximo, columna


# ----------------------- transformación de features -------------------------- #
@pytest.mark.parametrize("codigo, grupo", [
    ("428", "Circulatory"), ("250.02", "Diabetes"), ("486", "Respiratory"),
    ("820", "Injury"), ("V45", "Other"), ("?", "Missing"),
])
def test_icd9_group_agrupa_diagnosticos(codigo, grupo):
    assert icd9_group(codigo) == grupo


def test_build_features_devuelve_columnas_del_modelo_sin_nulos():
    X = build_features(pd.DataFrame([fila(), fila(race="?", age="?")]))
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.isna().sum().sum() == 0
    assert X.loc[1, "race"] == "Unknown"


def test_build_features_calcula_variables_derivadas():
    r = build_features(pd.DataFrame([fila()])).iloc[0]
    assert r.age_years == 75
    assert r.service_utilization == 3          # 0 ambulatorias + 1 urgencia + 2 hospitalizaciones
    assert r.num_meds_changed == 1             # insulina "Up"
    assert r.num_meds_active == 2              # insulina y metformina
    assert (r.med_change, r.diabetes_med) == (1, 1)
    assert r.medical_specialty_group == "Unknown"


def test_build_features_acepta_tipos_numericos_como_los_envia_la_api():
    X = build_features(pd.DataFrame([fila(time_in_hospital=5, number_inpatient=0)]))
    assert X.iloc[0].time_in_hospital == 5


def test_objetivo_y_exclusiones():
    df = pd.DataFrame([fila(), fila(readmitted="NO", discharge_disposition_id="11"),
                       fila(gender="Unknown/Invalid")])
    assert build_target(df).tolist() == [1, 0, 1]
    assert exclusion_reason(df).tolist() == [None, "expired_or_hospice", "invalid_gender"]


def test_particion_es_deterministica_y_proporcional():
    assert assign_split(123) == assign_split("123")
    proporcion = pd.Series([assign_split(i) for i in range(5000)]).value_counts(normalize=True)
    assert math.isclose(proporcion["train"], 0.70, abs_tol=0.03)
