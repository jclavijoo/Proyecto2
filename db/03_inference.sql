-- =========================================================================
-- Capa INFERENCE: cada solicitud a /predict queda registrada. Insumo para
-- reentrenamiento, auditoría y monitoreo de deriva.
-- =========================================================================
CREATE SCHEMA IF NOT EXISTS inference;

CREATE TABLE IF NOT EXISTS inference.prediction_logs (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id          TEXT        NOT NULL UNIQUE,      -- identificador de la solicitud
    requested_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    input_payload       JSONB       NOT NULL,             -- datos recibidos por la API
    features            JSONB,                            -- características calculadas
    prediction          SMALLINT,
    prediction_label    TEXT,
    probability         DOUBLE PRECISION,
    decision_threshold  DOUBLE PRECISION,
    model_name          TEXT,
    model_version       TEXT,
    model_alias         TEXT,
    model_run_id        TEXT,
    latency_ms          DOUBLE PRECISION,                 -- tiempo de respuesta
    status              TEXT NOT NULL CHECK (status IN ('success', 'error')),
    error_message       TEXT,
    client_host         TEXT,
    used_for_training   BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS ix_pred_requested_at ON inference.prediction_logs (requested_at);
CREATE INDEX IF NOT EXISTS ix_pred_model_version ON inference.prediction_logs (model_version);

-- Vista de apoyo para monitoreo
CREATE OR REPLACE VIEW inference.v_hourly_summary AS
SELECT date_trunc('hour', requested_at) AS hour,
       model_version,
       count(*)                                   AS requests,
       count(*) FILTER (WHERE status = 'error')   AS errors,
       avg(probability)                           AS avg_probability,
       avg(prediction::float)                     AS positive_rate,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms
FROM inference.prediction_logs
GROUP BY 1, 2;
