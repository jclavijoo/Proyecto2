# Guion de sustentación (máximo 10 minutos)

| Min | Bloque | Qué mostrar |
|---|---|---|
| 0:00–1:00 | Arquitectura | Diagrama del enunciado; en Rancher, el clúster y sus nodos, y luego **Workloads** del namespace `mlops` (todo en Running) |
| 1:00–2:00 | Kubernetes | Un Deployment con requests/limits y probes, un Secret, el ConfigMap del pipeline y las imágenes en DockerHub |
| 2:00–3:30 | Airflow | Grafo del DAG, ejecución de un lote, logs de `load_raw_batch` y `validate_raw_quality`, y una reejecución que no duplica datos |
| 3:30–4:30 | Datos | Consultas con `kubectl exec postgres-data-0 -- psql` (README §7): lotes de 15.000, estados en RAW, partición en CLEAN y la justificación de las transformaciones |
| 4:30–6:00 | MLflow | Runs comparados, métrica F1 (por qué no accuracy), artefactos (matriz de confusión), versiones y alias `@champion` / `@previous-champion` |
| 6:00–7:00 | Inferencia | Streamlit con un ejemplo, un caso inválido y la fila nueva en `inference.prediction_logs`; luego promover otro modelo y mostrar que la API cambia de versión sola |
| 7:00–9:00 | Carga + observabilidad | Locust subiendo usuarios con Grafana al lado: RPS, p95 y CPU. Señalar el punto donde las RPS se aplanan y el p95 se dispara (límite con 1 CPU). Escalar a 3 réplicas (`kubectl get pods -o wide` muestra el nodo de cada una) y comparar |
| 9:00–10:00 | Cierre | Dificultades encontradas (ver README §10.7) y mejoras futuras: reentrenar con los logs de inferencia, monitoreo de deriva, HPA |
