-- =========================================================================
-- Capa CLEAN: datos limpios + ingeniería de características. Fuente única
-- del entrenamiento. Las columnas deben coincidir con
-- airflow/dags/diabetes_pipeline/features.py (FEATURE_COLUMNS).
-- =========================================================================
CREATE SCHEMA IF NOT EXISTS clean;

CREATE TABLE IF NOT EXISTS clean.diabetes_clean (
    encounter_id             BIGINT PRIMARY KEY,
    patient_nbr              BIGINT  NOT NULL,
    raw_id                   BIGINT  NOT NULL,           -- trazabilidad hacia RAW
    batch_id                 TEXT    NOT NULL,
    -- categóricas
    race                     TEXT    NOT NULL,
    gender                   TEXT    NOT NULL,
    admission_type           TEXT    NOT NULL,
    discharge_group          TEXT    NOT NULL,
    admission_source_group   TEXT    NOT NULL,
    medical_specialty_group  TEXT    NOT NULL,
    diag_1_group             TEXT    NOT NULL,
    diag_2_group             TEXT    NOT NULL,
    diag_3_group             TEXT    NOT NULL,
    max_glu_serum            TEXT    NOT NULL,
    a1c_result               TEXT    NOT NULL,
    insulin                  TEXT    NOT NULL,
    metformin                TEXT    NOT NULL,
    -- numéricas
    age_years                REAL     NOT NULL,
    time_in_hospital         SMALLINT NOT NULL,
    num_lab_procedures       SMALLINT NOT NULL,
    num_procedures           SMALLINT NOT NULL,
    num_medications          SMALLINT NOT NULL,
    number_outpatient        SMALLINT NOT NULL,
    number_emergency         SMALLINT NOT NULL,
    number_inpatient         SMALLINT NOT NULL,
    number_diagnoses         SMALLINT NOT NULL,
    service_utilization      SMALLINT NOT NULL,
    num_meds_changed         SMALLINT NOT NULL,
    num_meds_active          SMALLINT NOT NULL,
    med_change               SMALLINT NOT NULL,
    diabetes_med             SMALLINT NOT NULL,
    -- objetivo
    readmitted               TEXT     NOT NULL,          -- valor original (NO, >30, <30)
    readmitted_30            SMALLINT NOT NULL CHECK (readmitted_30 IN (0, 1)),
    -- control
    split                    TEXT CHECK (split IN ('train', 'val', 'test')),
    transform_version        TEXT        NOT NULL,
    processed_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    split_assigned_at        TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_clean_split ON clean.diabetes_clean (split);
CREATE INDEX IF NOT EXISTS ix_clean_patient ON clean.diabetes_clean (patient_nbr);
CREATE INDEX IF NOT EXISTS ix_clean_batch ON clean.diabetes_clean (batch_id);
