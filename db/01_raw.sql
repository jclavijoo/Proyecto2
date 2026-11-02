-- =========================================================================
-- Capa RAW: datos tal como llegan del archivo fuente (sin transformaciones
-- destructivas). Nunca se sobrescribe: solo se insertan filas nuevas y se
-- actualiza el estado de procesamiento.
-- Idempotente: puede ejecutarse varias veces.
-- =========================================================================
CREATE SCHEMA IF NOT EXISTS raw;

-- Control de lotes de carga (uno por ejecución del DAG)
CREATE TABLE IF NOT EXISTS raw.load_batches (
    batch_id         TEXT PRIMARY KEY,                 -- <archivo>_<fila_ini>_<fila_fin>
    source_file      TEXT        NOT NULL,
    source_sha256    TEXT,
    offset_start     INTEGER     NOT NULL,             -- filas del archivo ya cargadas antes del lote
    offset_end       INTEGER     NOT NULL,
    rows_read        INTEGER     NOT NULL,
    rows_inserted    INTEGER     NOT NULL DEFAULT 0,
    rows_duplicated  INTEGER     NOT NULL DEFAULT 0,
    status           TEXT        NOT NULL CHECK (status IN ('LOADING', 'COMPLETED', 'FAILED')),
    dag_run_id       TEXT,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at      TIMESTAMPTZ,
    CONSTRAINT batch_size_max CHECK (rows_read <= 15000)
);
CREATE INDEX IF NOT EXISTS ix_load_batches_source ON raw.load_batches (source_file, status);
CREATE INDEX IF NOT EXISTS ix_load_batches_run ON raw.load_batches (dag_run_id);

-- Registros crudos: el contenido original de cada fila se guarda completo en JSONB
CREATE TABLE IF NOT EXISTS raw.diabetes_raw (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id           TEXT        NOT NULL REFERENCES raw.load_batches (batch_id),
    load_timestamp     TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_file        TEXT        NOT NULL,           -- origen del dato
    source_row_number  INTEGER     NOT NULL,           -- fila dentro del archivo
    encounter_id       TEXT,                           -- identificador del registro
    row_hash           CHAR(64)    NOT NULL UNIQUE,    -- control de duplicados
    payload            JSONB       NOT NULL,           -- fila original sin transformar
    status             TEXT        NOT NULL DEFAULT 'NEW'
                       CHECK (status IN ('NEW', 'VALID', 'REJECTED', 'DUPLICATE',
                                         'PROCESSED', 'EXCLUDED')),
    status_reason      TEXT,
    status_updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    dag_run_id         TEXT
);
CREATE INDEX IF NOT EXISTS ix_raw_status ON raw.diabetes_raw (status);
CREATE INDEX IF NOT EXISTS ix_raw_batch ON raw.diabetes_raw (batch_id);
CREATE INDEX IF NOT EXISTS ix_raw_encounter ON raw.diabetes_raw (encounter_id);

-- Resultado de cada validación de calidad
CREATE TABLE IF NOT EXISTS raw.quality_checks (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    checked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    dag_run_id      TEXT,
    rows_checked    INTEGER NOT NULL,
    rows_valid      INTEGER NOT NULL,
    rows_rejected   INTEGER NOT NULL,
    rows_duplicate  INTEGER NOT NULL,
    passed          BOOLEAN NOT NULL,
    details         JSONB
);
