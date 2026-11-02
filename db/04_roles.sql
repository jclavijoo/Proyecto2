-- =========================================================================
-- Usuario restringido para la API: solo puede INSERTAR en la capa de
-- inferencia (principio de mínimo privilegio). Airflow usa el usuario dueño.
-- Variables psql: :api_user, :api_password (ver run_migrations.sh)
-- =========================================================================
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'api_user', :'api_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'api_user') \gexec

SELECT format('ALTER ROLE %I WITH LOGIN PASSWORD %L', :'api_user', :'api_password') \gexec
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'api_user') \gexec
SELECT format('GRANT USAGE ON SCHEMA inference TO %I', :'api_user') \gexec
SELECT format('GRANT INSERT ON inference.prediction_logs TO %I', :'api_user') \gexec
SELECT format('GRANT USAGE ON ALL SEQUENCES IN SCHEMA inference TO %I', :'api_user') \gexec
SELECT format('REVOKE ALL ON SCHEMA raw, clean FROM %I', :'api_user') \gexec
