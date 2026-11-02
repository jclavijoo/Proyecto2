"""Ingesta: validación del archivo fuente y carga incremental a la capa RAW.

Estrategia de idempotencia:
  * El offset de lectura se deriva de los lotes COMPLETED en raw.load_batches,
    no de un contador en memoria -> si el DAG falla, la siguiente ejecución
    retoma donde quedó.
  * Si la misma ejecución del DAG (dag_run_id) se reintenta, se reutiliza el
    lote que ya cargó en vez de cargar uno nuevo.
  * Cada fila lleva un row_hash (SHA-256 del contenido) con restricción UNIQUE:
    ON CONFLICT DO NOTHING evita duplicados aunque el archivo se cargue dos veces.
  * Todo ocurre en UNA transacción con un advisory lock: o se guarda el lote
    completo (filas + registro de control) o no se guarda nada.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import psycopg2.extras

from diabetes_pipeline import config
from diabetes_pipeline.db import get_conn
from diabetes_pipeline.features import missing_required_columns

log = logging.getLogger(__name__)
LOAD_LOCK_ID = 727_001  # advisory lock para que nunca haya dos cargas a la vez


def _download(url: str, target: Path) -> None:
    import requests  # disponible en la imagen de Airflow

    log.info("Descargando archivo fuente desde %s", url)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.rename(target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_source_file() -> dict:
    """Tarea 1: el archivo existe, no está vacío y tiene las columnas esperadas."""
    path = Path(config.SOURCE_FILE_PATH)
    if not path.exists():
        if not config.SOURCE_FILE_URL:
            raise FileNotFoundError(f"No existe {path} y SOURCE_FILE_URL está vacío")
        try:
            _download(config.SOURCE_FILE_URL, path)
        except PermissionError:  # /opt/airflow/data no escribible -> /tmp
            path = Path("/tmp/data") / path.name
            if not path.exists():
                _download(config.SOURCE_FILE_URL, path)

    if path.stat().st_size == 0:
        raise ValueError(f"El archivo {path} está vacío")

    header = pd.read_csv(path, nrows=5, dtype=str, keep_default_na=False)
    missing = missing_required_columns(header.columns)
    if missing:
        raise ValueError(f"El archivo no tiene las columnas requeridas: {missing}")

    with open(path, "rb") as fh:
        total_rows = sum(1 for _ in fh) - 1  # menos el encabezado

    info = {
        "path": str(path),
        "source_file": path.name,
        "size_bytes": path.stat().st_size,
        "total_rows": total_rows,
        "sha256": _sha256(path),
        "columns": len(header.columns),
    }
    log.info("Archivo fuente OK: %s", info)
    return info


def _row_hash(record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_next_batch(source: dict, batch_size: int, dag_run_id: str) -> dict:
    """Tarea 2: carga el siguiente lote (<= 15.000 filas) a raw.diabetes_raw."""
    batch_size = min(int(batch_size), config.MAX_BATCH_SIZE)
    source_file = source["source_file"]

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (LOAD_LOCK_ID,))

        # Reintento de la misma ejecución -> devolver el lote ya cargado
        cur.execute(
            """SELECT batch_id, offset_start, offset_end, rows_read, rows_inserted
               FROM raw.load_batches WHERE dag_run_id = %s AND status = 'COMPLETED'""",
            (dag_run_id,),
        )
        existing = cur.fetchone()
        if existing:
            log.info("La ejecución %s ya cargó el lote %s; se reutiliza", dag_run_id, existing[0])
            return {"batch_id": existing[0], "offset_start": existing[1],
                    "offset_end": existing[2], "rows_read": existing[3],
                    "rows_inserted": existing[4], "reused": True, "exhausted": False}

        cur.execute(
            """SELECT COALESCE(MAX(offset_end), 0) FROM raw.load_batches
               WHERE source_file = %s AND status = 'COMPLETED'""",
            (source_file,),
        )
        offset = int(cur.fetchone()[0])

        if offset >= source["total_rows"]:
            log.info("No quedan filas por cargar (offset=%s, total=%s)", offset, source["total_rows"])
            return {"batch_id": None, "rows_read": 0, "rows_inserted": 0,
                    "offset_start": offset, "offset_end": offset,
                    "reused": False, "exhausted": True}

        df = pd.read_csv(
            source["path"], dtype=str, keep_default_na=False,  # sin transformar: '?' se conserva
            skiprows=range(1, offset + 1), nrows=batch_size,
        )
        offset_end = offset + len(df)
        batch_id = f"{Path(source_file).stem}_{offset + 1:06d}_{offset_end:06d}"

        cur.execute(
            """INSERT INTO raw.load_batches
                 (batch_id, source_file, source_sha256, offset_start, offset_end,
                  rows_read, rows_inserted, rows_duplicated, status, dag_run_id)
               VALUES (%s, %s, %s, %s, %s, %s, 0, 0, 'LOADING', %s)
               ON CONFLICT (batch_id) DO UPDATE
                 SET status = 'LOADING', dag_run_id = EXCLUDED.dag_run_id, started_at = now()""",
            (batch_id, source_file, source["sha256"], offset, offset_end, len(df), dag_run_id),
        )

        records = df.to_dict(orient="records")
        rows = [
            (batch_id, source_file, offset + i + 1, rec.get("encounter_id"),
             _row_hash(rec), psycopg2.extras.Json(rec), dag_run_id)
            for i, rec in enumerate(records)
        ]
        inserted = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO raw.diabetes_raw
                 (batch_id, source_file, source_row_number, encounter_id,
                  row_hash, payload, dag_run_id)
               VALUES %s
               ON CONFLICT (row_hash) DO NOTHING
               RETURNING id""",
            rows, page_size=1000, fetch=True,
        )
        n_inserted = len(inserted)

        cur.execute(
            """UPDATE raw.load_batches
               SET status = 'COMPLETED', rows_inserted = %s, rows_duplicated = %s,
                   finished_at = now()
               WHERE batch_id = %s""",
            (n_inserted, len(df) - n_inserted, batch_id),
        )

    result = {"batch_id": batch_id, "offset_start": offset, "offset_end": offset_end,
              "rows_read": len(df), "rows_inserted": n_inserted,
              "reused": False, "exhausted": False}
    log.info("Lote cargado: %s", result)
    return result


def pending_rows() -> int:
    """Filas RAW que aún no terminan el flujo (NEW o VALID)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw.diabetes_raw WHERE status IN ('NEW', 'VALID')")
        return int(cur.fetchone()[0])

