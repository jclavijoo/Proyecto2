#!/bin/sh
# Ejecutado por el Job data-db-migrate. Aplica los .sql en orden (idempotentes).
set -eu
echo "Esperando a PostgreSQL en ${PGHOST}:${PGPORT:-5432}..."
until pg_isready -h "$PGHOST" -p "${PGPORT:-5432}" -U "$PGUSER" -d "$PGDATABASE" >/dev/null 2>&1; do
  sleep 2
done
for f in /migrations/*.sql; do
  echo ">> Aplicando $f"
  psql -v ON_ERROR_STOP=1 -v api_user="$API_DB_USER" -v api_password="$API_DB_PASSWORD" -f "$f"
done
echo "Migraciones aplicadas correctamente."
