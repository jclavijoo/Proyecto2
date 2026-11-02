"""Acceso a PostgreSQL (capa de datos) con psycopg2 puro.

Se evita depender de la versión de SQLAlchemy de la imagen de Airflow.
"""
from __future__ import annotations

import contextlib
from typing import Iterator, Sequence

import pandas as pd
import psycopg2
import psycopg2.extras

from diabetes_pipeline import config


@contextlib.contextmanager
def get_conn() -> Iterator[psycopg2.extensions.connection]:
    """Conexión transaccional: commit si todo sale bien, rollback si falla."""
    conn = psycopg2.connect(connect_timeout=10, **config.DATA_DB)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_df(cur, query: str, params: Sequence | dict | None = None) -> pd.DataFrame:
    cur.execute(query, params)
    columns = [d[0] for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=columns)


def read_df(query: str, params: Sequence | dict | None = None) -> pd.DataFrame:
    with get_conn() as conn, conn.cursor() as cur:
        return fetch_df(cur, query, params)


def check_schema() -> None:
    """Falla rápido si el Job de migración no creó las tablas."""
    required = ["raw.load_batches", "raw.diabetes_raw", "raw.quality_checks",
                "clean.diabetes_clean", "inference.prediction_logs"]
    with get_conn() as conn, conn.cursor() as cur:
        missing = []
        for table in required:
            cur.execute("SELECT to_regclass(%s)", (table,))
            if cur.fetchone()[0] is None:
                missing.append(table)
    if missing:
        raise RuntimeError(
            f"Faltan tablas {missing}. Ejecute el Job 'data-db-migrate' (ver README)."
        )
