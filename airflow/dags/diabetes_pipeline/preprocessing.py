"""Transformación RAW -> CLEAN y partición train/val/test.

* preprocess_valid_rows: toma las filas VALID de RAW, aplica la ingeniería de
  características de features.py, hace UPSERT en
  clean.diabetes_clean y marca las filas RAW como PROCESSED / EXCLUDED, todo en
  una sola transacción (no queda estado intermedio inconsistente).
* assign_splits: asigna la partición a las filas nuevas de CLEAN de forma
  determinística y agrupada por paciente.
"""
from __future__ import annotations

import logging

import pandas as pd
import psycopg2.extras

from diabetes_pipeline import config
from diabetes_pipeline.db import fetch_df, get_conn
from diabetes_pipeline.features import (
    FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN, TARGET_RAW_COLUMN,
    TRANSFORM_VERSION, assign_split, build_features, build_target, exclusion_reason,
)

log = logging.getLogger(__name__)

CLEAN_COLUMNS = (["encounter_id", "patient_nbr", "raw_id", "batch_id"] + FEATURE_COLUMNS
                 + [TARGET_RAW_COLUMN, TARGET_COLUMN, "transform_version"])


def preprocess_valid_rows() -> dict:
    with get_conn() as conn, conn.cursor() as cur:
        valid = fetch_df(
            cur, "SELECT id, batch_id, payload FROM raw.diabetes_raw "
                 "WHERE status = 'VALID' ORDER BY id")
        if valid.empty:
            return {"rows_in": 0, "rows_clean": 0, "rows_excluded": 0}

        data = pd.DataFrame(list(valid["payload"]), index=valid.index)
        reason = exclusion_reason(data)
        keep = reason.isna()

        features = build_features(data.loc[keep])
        clean = features.copy()
        clean.insert(0, "batch_id", valid.loc[keep, "batch_id"])
        clean.insert(0, "raw_id", valid.loc[keep, "id"].astype(int))
        clean.insert(0, "patient_nbr", pd.to_numeric(data.loc[keep, "patient_nbr"]).astype(int))
        clean.insert(0, "encounter_id", pd.to_numeric(data.loc[keep, "encounter_id"]).astype(int))
        clean[TARGET_RAW_COLUMN] = data.loc[keep, TARGET_RAW_COLUMN].astype(str).str.strip()
        clean[TARGET_COLUMN] = build_target(data.loc[keep])
        clean["transform_version"] = TRANSFORM_VERSION

        records = [
            tuple(v.item() if hasattr(v, "item") else v for v in row)
            for row in clean[CLEAN_COLUMNS].itertuples(index=False, name=None)
        ]
        update_set = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in CLEAN_COLUMNS if c != "encounter_id"
        )
        if records:
            psycopg2.extras.execute_values(
                cur,
                f"""INSERT INTO clean.diabetes_clean ({", ".join(CLEAN_COLUMNS)})
                    VALUES %s
                    ON CONFLICT (encounter_id) DO UPDATE
                    SET {update_set}, processed_at = now()""",
                records, page_size=1000,
            )

        status_updates = [
            (int(rid), "PROCESSED" if r is None else "EXCLUDED", r)
            for rid, r in zip(valid["id"], reason)
        ]
        psycopg2.extras.execute_values(
            cur,
            """UPDATE raw.diabetes_raw AS t
               SET status = v.status, status_reason = v.reason, status_updated_at = now()
               FROM (VALUES %s) AS v(id, status, reason)
               WHERE t.id = v.id""",
            status_updates, page_size=2000,
        )

    summary = {
        "rows_in": int(len(valid)),
        "rows_clean": int(keep.sum()),
        "rows_excluded": int((~keep).sum()),
        "exclusion_reasons": reason.dropna().value_counts().to_dict(),
        "positive_rate_batch": round(float(clean[TARGET_COLUMN].mean()), 4) if len(clean) else None,
        "transform_version": TRANSFORM_VERSION,
        "numeric_features": len(NUMERIC_FEATURES),
        "total_features": len(FEATURE_COLUMNS),
    }
    log.info("Preprocesamiento: %s", summary)
    return summary


def assign_splits() -> dict:
    with get_conn() as conn, conn.cursor() as cur:
        pending = fetch_df(
            cur, "SELECT encounter_id, patient_nbr FROM clean.diabetes_clean WHERE split IS NULL")
        if not pending.empty:
            pending["split"] = pending["patient_nbr"].map(
                lambda p: assign_split(p, config.TRAIN_RATIO, config.VAL_RATIO))
            psycopg2.extras.execute_values(
                cur,
                """UPDATE clean.diabetes_clean AS t
                   SET split = v.split, split_assigned_at = now()
                   FROM (VALUES %s) AS v(encounter_id, split)
                   WHERE t.encounter_id = v.encounter_id""",
                [(int(e), s) for e, s in zip(pending["encounter_id"], pending["split"])],
                page_size=2000,
            )
        stats = fetch_df(
            cur, f"""SELECT split, count(*) AS rows, sum({TARGET_COLUMN}) AS positives,
                            count(DISTINCT patient_nbr) AS patients
                     FROM clean.diabetes_clean GROUP BY split""")

    by_split = {
        r.split: {"rows": int(r.rows), "positives": int(r.positives),
                  "patients": int(r.patients),
                  "positive_rate": round(int(r.positives) / int(r.rows), 4)}
        for r in stats.itertuples()
    }
    for name in ("train", "val", "test"):
        if name not in by_split or by_split[name]["positives"] < config.MIN_POSITIVES_PER_SPLIT:
            raise ValueError(
                f"La partición '{name}' no tiene suficientes positivos: {by_split.get(name)}")

    result = {"newly_assigned": int(len(pending)), "splits": by_split}
    log.info("Particiones: %s", result)
    return result
