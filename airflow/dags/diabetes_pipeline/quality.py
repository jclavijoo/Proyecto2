"""Validación básica de calidad sobre las filas RAW con estado NEW.

Reglas por fila (las que fallan quedan REJECTED con su motivo):
  * encounter_id y patient_nbr numéricos.
  * readmitted dentro de {NO, >30, <30}.
  * Columnas numéricas enteras y dentro de NUMERIC_RANGES (features.py), los
    mismos límites que valida la API (por ejemplo time_in_hospital entre 1 y 14).
  * encounter_id no repetido (contra lo ya validado y dentro del propio lote) -> DUPLICATE.

Regla global: si la proporción de rechazados supera MAX_REJECT_RATIO la tarea
FALLA (error visible en Airflow), los estados NO se modifican (rollback) y el
reporte queda guardado en raw.quality_checks para auditoría.
"""
from __future__ import annotations

import logging

import pandas as pd
import psycopg2.extras

from diabetes_pipeline import config
from diabetes_pipeline.db import fetch_df, get_conn
from diabetes_pipeline.features import VALID_TARGET_VALUES, numeric_range_violations

log = logging.getLogger(__name__)

def evaluate_rows(df: pd.DataFrame) -> pd.Series:
    """Devuelve el motivo de rechazo por fila (None = válida)."""
    reason = pd.Series([None] * len(df), index=df.index, dtype=object)

    def flag(mask: pd.Series, text: str) -> None:
        reason[mask & reason.isna()] = text

    for col in ("encounter_id", "patient_nbr"):
        flag(pd.to_numeric(df[col], errors="coerce").isna(), f"{col}_not_numeric")
    flag(~df["readmitted"].astype(str).str.strip().isin(VALID_TARGET_VALUES), "invalid_target")
    ranges = numeric_range_violations(df)
    return reason.where(reason.notna(), ranges)


def validate_pending(dag_run_id: str) -> dict:
    with get_conn() as conn, conn.cursor() as cur:
        pending = fetch_df(
            cur, "SELECT id, encounter_id, payload FROM raw.diabetes_raw "
                 "WHERE status = 'NEW' ORDER BY id")
        if pending.empty:
            return {"rows_checked": 0, "rows_valid": 0, "rows_rejected": 0,
                    "rows_duplicate": 0, "passed": True}

        data = pd.DataFrame(list(pending["payload"]), index=pending.index)
        reason = evaluate_rows(data)

        # Duplicados contra filas ya aceptadas y dentro del lote pendiente
        cur.execute(
            "SELECT encounter_id FROM raw.diabetes_raw "
            "WHERE status IN ('VALID', 'PROCESSED', 'EXCLUDED') AND encounter_id = ANY(%s)",
            (pending["encounter_id"].dropna().unique().tolist(),),
        )
        already = {r[0] for r in cur.fetchall()}
        is_dup = pending["encounter_id"].isin(already) | pending["encounter_id"].duplicated()
        status = pd.Series("VALID", index=pending.index)
        status[reason.notna()] = "REJECTED"
        status[is_dup & reason.isna()] = "DUPLICATE"
        reason[is_dup & reason.isna()] = "duplicate_encounter_id"

        summary = {
            "rows_checked": int(len(pending)),
            "rows_valid": int((status == "VALID").sum()),
            "rows_rejected": int((status == "REJECTED").sum()),
            "rows_duplicate": int((status == "DUPLICATE").sum()),
            "reasons": reason.dropna().value_counts().to_dict(),
            "target_distribution": data["readmitted"].value_counts().to_dict(),
            "null_markers_per_column": {
                c: int((data[c].astype(str).str.strip() == "?").sum())
                for c in data.columns if (data[c].astype(str).str.strip() == "?").any()
            },
        }
        reject_ratio = summary["rows_rejected"] / summary["rows_checked"]
        summary["reject_ratio"] = round(reject_ratio, 4)
        summary["passed"] = reject_ratio <= config.MAX_REJECT_RATIO

        if summary["passed"]:
            updates = [(int(i), s, r) for i, s, r in zip(pending["id"], status, reason)]
            psycopg2.extras.execute_values(
                cur,
                """UPDATE raw.diabetes_raw AS t
                   SET status = v.status, status_reason = v.reason, status_updated_at = now()
                   FROM (VALUES %s) AS v(id, status, reason)
                   WHERE t.id = v.id""",
                updates, page_size=2000,
            )
            _save_report(cur, dag_run_id, summary)

    if not summary["passed"]:
        # Se guarda el reporte en una transacción aparte y se falla la tarea.
        with get_conn() as conn, conn.cursor() as cur:
            _save_report(cur, dag_run_id, summary)
        raise ValueError(
            f"Calidad insuficiente: {summary['reject_ratio']:.1%} de filas rechazadas "
            f"(máximo {config.MAX_REJECT_RATIO:.0%}). Detalle: {summary['reasons']}"
        )

    log.info("Validación de calidad: %s", summary)
    return summary


def _save_report(cur, dag_run_id: str, summary: dict) -> None:
    cur.execute(
        """INSERT INTO raw.quality_checks
             (dag_run_id, rows_checked, rows_valid, rows_rejected, rows_duplicate,
              passed, details)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (dag_run_id, summary["rows_checked"], summary["rows_valid"],
         summary["rows_rejected"], summary["rows_duplicate"], summary["passed"],
         psycopg2.extras.Json(summary)),
    )
