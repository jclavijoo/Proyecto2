"""Interfaz gráfica: consume EXCLUSIVAMENTE la API de inferencia (no MLflow ni la BD)."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://inference-api:8000").rstrip("/")
TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "10"))

AGES = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
        "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"]
MEDS = ["No", "Steady", "Up", "Down"]
EXTRA_MEDS = ["glipizide", "glyburide", "glimepiride", "pioglitazone", "rosiglitazone",
              "repaglinide", "nateglinide", "acarbose"]
ADMISSION_TYPES = {1: "1 - Emergency", 2: "2 - Urgent", 3: "3 - Elective", 4: "4 - Newborn",
                   5: "5 - Not Available", 6: "6 - NULL", 7: "7 - Trauma Center", 8: "8 - Not Mapped"}

EXAMPLES = {
    "Riesgo alto (múltiples hospitalizaciones)": {
        "race": "Caucasian", "gender": "Female", "age": "[70-80)", "admission_type_id": 1,
        "discharge_disposition_id": 3, "admission_source_id": 7, "time_in_hospital": 9,
        "medical_specialty": "InternalMedicine", "num_lab_procedures": 65, "num_procedures": 2,
        "num_medications": 25, "number_outpatient": 2, "number_emergency": 3,
        "number_inpatient": 5, "diag_1": "428", "diag_2": "250.02", "diag_3": "403",
        "number_diagnoses": 9, "max_glu_serum": ">300", "A1Cresult": ">8",
        "metformin": "No", "insulin": "Up", "change": "Ch", "diabetesMed": "Yes"},
    "Riesgo bajo (cirugía electiva)": {
        "race": "AfricanAmerican", "gender": "Male", "age": "[40-50)", "admission_type_id": 3,
        "discharge_disposition_id": 1, "admission_source_id": 1, "time_in_hospital": 2,
        "medical_specialty": "Orthopedics", "num_lab_procedures": 20, "num_procedures": 3,
        "num_medications": 8, "number_outpatient": 0, "number_emergency": 0,
        "number_inpatient": 0, "diag_1": "715", "diag_2": "250", "diag_3": "401",
        "number_diagnoses": 5, "max_glu_serum": "None", "A1Cresult": "Norm",
        "metformin": "Steady", "insulin": "No", "change": "No", "diabetesMed": "Yes"},
    "Caso inválido (para ver errores de validación)": {
        "race": "Caucasian", "gender": "Female", "age": "[60-70)", "admission_type_id": 1,
        "discharge_disposition_id": 1, "admission_source_id": 7, "time_in_hospital": 20,
        "medical_specialty": "?", "num_lab_procedures": 30, "num_procedures": 0,
        "num_medications": 10, "number_outpatient": 0, "number_emergency": 0,
        "number_inpatient": 0, "diag_1": "", "diag_2": "?", "diag_3": "?",
        "number_diagnoses": 0, "max_glu_serum": "None", "A1Cresult": "None",
        "metformin": "No", "insulin": "No", "change": "No", "diabetesMed": "No"},
}
DEFAULTS = EXAMPLES["Riesgo bajo (cirugía electiva)"]

st.set_page_config(page_title="Readmisión hospitalaria - Diabetes", page_icon="🏥", layout="wide")

# Estado inicial del formulario (los widgets leen su valor desde session_state)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
for _m in EXTRA_MEDS:
    st.session_state.setdefault(f"med_{_m}", "No")


def api_get(path: str):
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=TIMEOUT)
        return resp.status_code, resp.json()
    except requests.RequestException as exc:
        return None, str(exc)


# ------------------------------- barra lateral ------------------------------ #
with st.sidebar:
    st.header("Estado del servicio")
    st.caption(f"API: `{API_URL}`")
    code, info = api_get("/model-info")
    if code == 200:
        if info["status"] == "loaded":
            st.success(f"Modelo **{info['model_name']}** v{info['model_version']} "
                       f"(@{info['model_alias']})")
            st.caption(f"Cargado: {info['loaded_at']}")
            tags = info.get("version_tags", {})
            if tags:
                st.json({k: tags[k] for k in sorted(tags)[:8]}, expanded=False)
        else:
            st.warning("La API está activa pero aún no hay modelo productivo. "
                       "Ejecute el DAG en Airflow.")
            if info.get("last_error"):
                st.caption(info["last_error"])
    else:
        st.error(f"No se pudo contactar la API: {info}")
    if st.button("Recargar modelo desde MLflow"):
        try:
            r = requests.post(f"{API_URL}/reload-model", timeout=60).json()
            st.info(r)
        except requests.RequestException as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Valores de ejemplo")
    example_name = st.selectbox("Escenario", list(EXAMPLES))
    if st.button("Cargar ejemplo", use_container_width=True):
        for key, value in EXAMPLES[example_name].items():
            st.session_state[key] = value
        st.rerun()




# ---------------------------------- formulario ------------------------------ #
st.title("🏥 Predicción de readmisión hospitalaria (< 30 días)")
st.write("Ingrese los datos del encuentro hospitalario. La predicción la genera el "
         "modelo productivo registrado en MLflow a través de la API de inferencia.")

with st.form("patient"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Paciente")
        race = st.selectbox("Raza", RACES, key="race")
        gender = st.selectbox("Género", ["Female", "Male"], key="gender")
        age = st.selectbox("Edad", AGES, key="age")
        st.subheader("Admisión")
        admission_type_id = st.selectbox(
            "Tipo de admisión", list(ADMISSION_TYPES), format_func=ADMISSION_TYPES.get, key="admission_type_id")
        discharge_disposition_id = st.number_input("ID disposición de alta", 1, 30,
                                                   key="discharge_disposition_id")
        admission_source_id = st.number_input("ID fuente de admisión", 1, 26,
                                              key="admission_source_id")
        medical_specialty = st.text_input("Especialidad médica ('?' si se desconoce)", key="medical_specialty")
    with c2:
        st.subheader("Estancia y procedimientos")
        # Sin min/max en estos campos para poder demostrar la validación de la API
        time_in_hospital = st.number_input("Días en el hospital (1-14)", step=1, key="time_in_hospital")
        num_lab_procedures = st.number_input("Procedimientos de laboratorio", 0, 200, key="num_lab_procedures")
        num_procedures = st.number_input("Procedimientos", 0, 10, key="num_procedures")
        num_medications = st.number_input("Número de medicamentos", 0, 100, key="num_medications")
        st.subheader("Visitas año anterior")
        number_outpatient = st.number_input("Ambulatorias", 0, 100, key="number_outpatient")
        number_emergency = st.number_input("Urgencias", 0, 100, key="number_emergency")
        number_inpatient = st.number_input("Hospitalizaciones", 0, 100, key="number_inpatient")
    with c3:
        st.subheader("Diagnósticos (ICD-9)")
        diag_1 = st.text_input("Diagnóstico principal", key="diag_1")
        diag_2 = st.text_input("Diagnóstico secundario", key="diag_2")
        diag_3 = st.text_input("Diagnóstico adicional", key="diag_3")
        number_diagnoses = st.number_input("Número de diagnósticos (1-20)", step=1, key="number_diagnoses")
        st.subheader("Laboratorio y medicación")
        max_glu_serum = st.selectbox("Glucosa sérica máx.", ["None", "Norm", ">200", ">300"],
                                     key="max_glu_serum")
        a1c = st.selectbox("A1C", ["None", "Norm", ">7", ">8"], key="A1Cresult")
        metformin = st.selectbox("Metformina", MEDS, key="metformin")
        insulin = st.selectbox("Insulina", MEDS, key="insulin")
        change = st.selectbox("¿Cambio de medicación?", ["No", "Ch"], key="change")
        diabetes_med = st.selectbox("¿Medicación para diabetes?", ["No", "Yes"], key="diabetesMed")

    with st.expander("Otros medicamentos para la diabetes"):
        cols = st.columns(4)
        other_meds = {m: cols[i % 4].selectbox(m, MEDS, key=f"med_{m}")
                      for i, m in enumerate(EXTRA_MEDS)}

    submitted = st.form_submit_button("Predecir", type="primary", use_container_width=True)

# ---------------------------------- predicción ------------------------------ #
if submitted:
    payload = {
        "race": race, "gender": gender, "age": age,
        "admission_type_id": int(admission_type_id),
        "discharge_disposition_id": int(discharge_disposition_id),
        "admission_source_id": int(admission_source_id),
        "time_in_hospital": int(time_in_hospital), "medical_specialty": medical_specialty or "?",
        "num_lab_procedures": int(num_lab_procedures), "num_procedures": int(num_procedures),
        "num_medications": int(num_medications), "number_outpatient": int(number_outpatient),
        "number_emergency": int(number_emergency), "number_inpatient": int(number_inpatient),
        "diag_1": diag_1.strip(), "diag_2": diag_2.strip() or "?", "diag_3": diag_3.strip() or "?",
        "number_diagnoses": int(number_diagnoses), "max_glu_serum": max_glu_serum,
        "A1Cresult": a1c, "metformin": metformin, "insulin": insulin,
        "change": change, "diabetesMed": diabetes_med, **other_meds,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT)
    except requests.RequestException as exc:
        st.error(f"No se pudo contactar la API: {exc}")
    else:
        if resp.status_code == 200:
            r = resp.json()
            risk = r["prediction"] == 1
            (st.error if risk else st.success)(
                "**Riesgo de readmisión en menos de 30 días**" if risk
                else "**Sin riesgo alto de readmisión temprana**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Probabilidad <30 días", f"{r['probability']:.1%}")
            m2.metric("Umbral", f"{r['decision_threshold']:.2f}")
            m3.metric("Modelo", f"v{r['model_version']} (@{r['model_alias']})")
            m4.metric("Tiempo de la API", f"{r['processing_time_ms']:.1f} ms")
            st.progress(min(max(r["probability"], 0.0), 1.0))
            st.caption(f"request_id: `{r['request_id']}` · modelo: {r['model_name']}")
        elif resp.status_code == 422:
            st.error("Datos inválidos. Corrija los siguientes campos:")
            for err in resp.json().get("detail", []):
                field = ".".join(str(p) for p in err.get("loc", [])[1:])
                st.write(f"- **{field}**: {err.get('msg')} (recibido: `{err.get('input')}`)")
        else:
            st.error(f"Error {resp.status_code}: {resp.text[:300]}")
