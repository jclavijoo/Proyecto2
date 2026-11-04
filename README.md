# Proyecto 2 – MLOps en Kubernetes (Rancher): readmisión hospitalaria de pacientes diabéticos

## Grupo 3

### Integrantes
* **Juan David Clavijo Ortiz**
* **Laura Sofía Rodríguez Pérez**
* **Joshua Alexander Valero Lozano**

## Link de Youtube: https://youtu.be/iDZpyuTl-RY

# Descripción

Arquitectura MLOps completa desplegada en Kubernetes administrado con **Rancher** sobre las VMs
Rocky, ingesta incremental orquestada con Airflow, capas RAW / CLEAN /
INFERENCE en PostgreSQL, experimentos y registro de modelos en MLflow (PostgreSQL + MinIO), API
de inferencia con FastAPI, interfaz en Streamlit, pruebas de carga con Locust y observabilidad
con Prometheus + Grafana.

**Problema:** predecir si un paciente diabético hospitalizado será readmitido en menos de
30 días (`readmitted == "<30"`), con el dataset *Diabetes 130-US hospitals (1999-2008)*.

## 1. Arquitectura

| Fase | Componente | Recurso en Kubernetes | Acceso |
|---|---|---|---|
| Entrenamiento | Airflow 3.1.8 (LocalExecutor) | Helm chart oficial 1.20.0 | NodePort 30080 |
| Entrenamiento | PostgreSQL de datos (`raw`, `clean`, `inference`) | StatefulSet + PVC + Job de migración | interno |
| Entrenamiento | MLflow 3.4.0 | Deployment | NodePort 30500 |
| Entrenamiento | PostgreSQL de metadatos de MLflow | StatefulSet + PVC | interno |
| Entrenamiento | MinIO (artefactos de MLflow) | StatefulSet + PVC | consola NodePort 30901 |
| Inferencia | API FastAPI | Deployment | NodePort 30800 |
| Inferencia | Streamlit | Deployment | NodePort 30501 |
| Inferencia | Locust | Deployment | NodePort 30089 |
| Observabilidad | Prometheus | Deployment + PVC + RBAC | NodePort 30909 |
| Observabilidad | Grafana | Deployment + PVC | NodePort 30300 |

Todos los contenedores tienen `requests`/`limits`, todos los servicios tienen probes, la
configuración está en ConfigMaps y las credenciales en Secrets. Todo vive en el namespace
`mlops`.

## 2. Estructura del repositorio

```
├── docker-compose.yaml         Desarrollo local + build/push de imágenes (como Niveles/3/Observabilidad)
├── .env.example                Variables de docker compose
├── airflow/                    Igual que Niveles/3/Airflow/proyecto_base
│   ├── Dockerfile, requirements.txt
│   ├── dags/                   DAG + módulos del pipeline (incluye features.py)
│   ├── data/                   Aquí va Diabetes.csv (se copia a la imagen)
│   └── values/values-proyecto.yaml
├── mlflow/                     Dockerfile del servidor + creación del bucket en MinIO
├── api/                        API FastAPI: Dockerfile, requirements.txt, main.py, features.py, ...
├── streamlit/                  Interfaz gráfica: Dockerfile, requirements.txt, app.py
├── locust/                     Dockerfile, requirements.txt, locustfile.py (como Niveles/3/locust)
├── observabilidad/             
│   ├── prometheus.yml          (docker compose)   prometheus-k8s.yml (Kubernetes)
│   └── grafana/provisioning/ y grafana/dashboards/mlops-inference-api.json
├── db/                         Scripts SQL de creación de tablas (idempotentes)
├── k8s/                        Manifiestos: kubectl apply -f k8s/
├── tests/                      Pruebas con pytest (como Niveles/3/CICD/unit_testing)
├── .github/workflows/          CI: pruebas + publicación en DockerHub (plantilla del curso)
├── docs/guion_sustentacion.md
└── notebooks/                  EDA
```


**3.2 StorageClass (necesaria para los PVC).**

```bash
kubectl get storageclass
```

- Clúster `local` (K3s): ya trae `local-path (default)`. No hay que hacer nada.
- Clúster Custom **RKE2**: **no trae ninguna**, y sin ella todos los PVC quedan en `Pending`.
  Instale el aprovisionador de Rancher y márquelo por defecto:

```bash
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.31/deploy/local-path-storage.yaml
kubectl patch storageclass local-path -p '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

**3.3 Firewall.** Los NodePort (30000-32767) ya están abiertos si siguió la guía de nodos RKE2.

## 4. Construir y publicar las imágenes (DockerHub)

Cada imagen se construye dentro de su carpeta, como en los ejercicios:

```bash
docker login

cd airflow
curl -L "https://drive.usercontent.google.com/download?id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC&export=download&confirm=t" -o data/Diabetes.csv
docker build --build-arg AIRFLOW_BASE_TAG=3.1.8 -t su_usuario/mlops-p2-airflow:0.1.0 .
docker push su_usuario/mlops-p2-airflow:0.1.0
cd ..

cd mlflow    && docker build -t su_usuario/mlops-p2-mlflow:0.1.0 .    && docker push su_usuario/mlops-p2-mlflow:0.1.0    && cd ..
cd api       && docker build -t su_usuario/mlops-p2-api:0.1.0 .       && docker push su_usuario/mlops-p2-api:0.1.0       && cd ..
cd streamlit && docker build -t su_usuario/mlops-p2-streamlit:0.1.0 . && docker push su_usuario/mlops-p2-streamlit:0.1.0 && cd ..
cd locust    && docker build -t su_usuario/mlops-p2-locust:0.1.0 .    && docker push su_usuario/mlops-p2-locust:0.1.0    && cd ..
```

Otra opción para las cuatro últimas imágenes: `cp .env.example .env`, poner su usuario en
`DOCKERHUB_USER` y ejecutar `docker compose build` y luego `docker compose push`.

Si `airflow/data/Diabetes.csv` no está en la imagen, la primera tarea del DAG lo descarga.
Cuando cambie el código, suba el tag (`0.1.1`, `0.1.2`…) en el build y en los manifiestos,
como indica la guía de Airflow.

Si construye en un Mac con chip Apple, agregue `--platform linux/amd64` al `docker build`, porque las VMs son x86.

## 5. Desplegar en Kubernetes

```bash
# 1. Su usuario de DockerHub en los manifiestos y en los values de Airflow
sed -i 's/DOCKERHUB_USER/su_usuario/g' k8s/*.yaml airflow/values/values-proyecto.yaml

# 2. Cambie las contraseñas en k8s/01-secrets.yaml (iguales a las del .env)

# 3. Aplicar todo
kubectl apply -f k8s/
kubectl config set-context --current --namespace=mlops    # para no escribir -n mlops
kubectl get pods -w                                       # esperar Running / Completed

# 4. Airflow con Helm (guía Niveles/3/Airflow)
helm repo add apache-airflow https://airflow.apache.org && helm repo update
helm upgrade --install airflow apache-airflow/airflow --version 1.20.0 \
  --namespace mlops -f airflow/values/values-proyecto.yaml --wait --timeout 20m
```

Alternativa sin `kubectl` en la VM: **Import YAML** en Rancher, pegando los archivos de `k8s/`
en orden (primero `00-namespace.yaml` y `01-secrets.yaml`). Todo queda visible en Rancher en
**Workloads**, **Service Discovery** y **Storage > PersistentVolumeClaims**.

**¿Por qué no se usa la salida de `kompose` directamente?** Al ejecutar
`kompose convert -f docker-compose.yaml` sobre este proyecto pasa lo siguiente:

- No genera Secrets: las contraseñas quedan en texto plano dentro de los Deployments.
- Los archivos montados (SQL, `prometheus.yml`, provisioning de Grafana) se convierten en PVC
  vacíos.
- Las bases de datos salen como Deployments en vez de StatefulSets.
- No agrega probes.

Por eso `k8s/` contiene los manifiestos completados a mano, con la misma convención de nombres
de kompose (`<servicio>-deployment.yaml`, `<servicio>-service.yaml`).

Los ConfigMaps que vienen de archivos se generaron con
`kubectl create configmap ... --from-file --dry-run=client -o yaml`. Si cambia un `.sql`,
`prometheus-k8s.yml` o el dashboard, regenere el ConfigMap con el comando que está en el
encabezado de cada archivo.

## 6. Acceso a las interfaces

En un clúster **Custom RKE2**, use la IP de cualquier nodo:

| Interfaz | URL | Credenciales |
|---|---|---|
| Airflow | http://IP_NODO:30080 | admin / admin (usuario por defecto del chart) |
| MLflow | http://IP_NODO:30500 | – |
| MinIO | http://IP_NODO:30901 | `minio-secret` |
| API (Swagger) | http://IP_NODO:30800/docs | – |
| Streamlit | http://IP_NODO:30501 | – |
| Locust | http://IP_NODO:30089 | – |
| Prometheus | http://IP_NODO:30909 | – |
| Grafana | http://IP_NODO:30300 | `grafana-secret` |

En el clúster **`local`**, los NodePort quedan dentro del contenedor de Rancher, así que se usa
`port-forward` como en la guía:

```bash
kubectl port-forward --address 0.0.0.0 svc/streamlit 8501:8501
kubectl port-forward --address 0.0.0.0 svc/grafana 3000:3000
kubectl port-forward --address 0.0.0.0 svc/airflow-api-server 8080:8080
# ... uno por terminal (o con & al final)
```

## 7. Ejecución y validación

1. **Airflow:** active `diabetes_mlops_pipeline` y ejecútelo. Cada ejecución carga **un lote de
   15.000 filas**: el dataset (~101.766 filas) requiere 7 ejecuciones. Al terminar los datos,
   el DAG se detiene en `has_data_to_process`, salvo que se lance con `force_retrain=true`.

    <p align="center">
      <img src="Proyecto2/evidencias/Airflow.png" alt="Ejecución de Airflow" width="800">
    </p>

    <p align="center">
      <img src="Proyecto2/evidencias/Airflow2.png" alt="Ejecución de Airflow" width="800">
    </p>

2. **Capas de datos:**
   ```bash
   kubectl exec -it postgres-data-0 -- psql -U mlops -d diabetes -c "SELECT batch_id, rows_read, rows_inserted, status FROM raw.load_batches;"
   kubectl exec -it postgres-data-0 -- psql -U mlops -d diabetes -c "SELECT status, count(*) FROM raw.diabetes_raw GROUP BY 1;"
   kubectl exec -it postgres-data-0 -- psql -U mlops -d diabetes -c "SELECT split, count(*), avg(readmitted_30) FROM clean.diabetes_clean GROUP BY 1;"
   kubectl exec -it postgres-data-0 -- psql -U mlops -d diabetes -c "SELECT requested_at, prediction_label, probability, model_version, latency_ms FROM inference.prediction_logs ORDER BY id DESC LIMIT 5;"
   ```
3. **MLflow:** experimento `diabetes-readmission` (3 runs por ejecución) y modelo registrado
   `diabetes-readmission` con los alias `@champion`, `@challenger` y `@previous-champion`.
4. **API:**
   ```bash
   curl http://IP_NODO:30800/model-info
   curl -X POST http://IP_NODO:30800/predict -H "Content-Type: application/json" -d '{"gender":"Female","age":"[70-80)","admission_type_id":1,"discharge_disposition_id":1,"admission_source_id":7,"time_in_hospital":5,"num_lab_procedures":45,"num_procedures":1,"num_medications":18,"diag_1":"428","number_diagnoses":9,"insulin":"Up","diabetesMed":"Yes"}'
   ```
5. **Streamlit:** use "Cargar ejemplo" y "Predecir". El escenario "Caso inválido" muestra los
   errores de validación de la API campo por campo.

## 8. Pruebas de carga y observabilidad

Grafana abre directamente el dashboard **MLOps - API de inferencia**, que incluye: total de
solicitudes, solicitudes por segundo, latencia promedio, latencias p50/p95/p99, errores, tasa de
error, CPU y memoria del pod, *throttling* de CPU, predicciones por clase y modelo en producción.

**Cómo encontrar el punto de degradación:**

1. Con la API en 1 réplica (limit de 1 CPU), en Locust suba usuarios gradualmente (10, 25,
   50, 100…), o active el modo escalonado con `LOCUST_STEP_LOAD: "true"` en
   `k8s/locust-configmap.yaml` y reinicie Locust.
2. El límite operativo es el punto donde las RPS dejan de crecer mientras el p95 sigue subiendo.
   Debería coincidir con CPU cerca de 1 core y aumento del throttling.
3. Escale con `kubectl scale deploy/inference-api --replicas=3` y repita. Prometheus descubre
   las réplicas nuevas automáticamente. En un clúster multinodo, `kubectl get pods -o wide`
   muestra en qué nodo quedó cada una.

Locust corre dentro del clúster contra el Service de la API, así que la carga no pasa por un
port-forward, que sería un cuello de botella artificial.

## 9. Decisiones técnicas

### 9.1 Capas de datos

Una instancia de PostgreSQL con tres esquemas separados (`raw`, `clean`, `inference`) y una
instancia aparte para MLflow. Los esquemas permiten permisos distintos por capa: la API usa
el usuario `api_writer`, que solo puede hacer `INSERT` en `inference.prediction_logs` y
recibe *permission denied* sobre `raw` y `clean`. La BD de MLflow queda aislada porque su
esquema lo administra MLflow con sus propias migraciones.

- **RAW** (`raw.diabetes_raw`): la fila original completa en `JSONB`, sin transformar (los
  `?` se conservan). Incluye `batch_id`, `load_timestamp`, `source_file`, `source_row_number`,
  `row_hash` (único) y `status`. Nunca se sobrescribe: solo cambia el estado
  (NEW → VALID / REJECTED / DUPLICATE → PROCESSED / EXCLUDED).
- **CLEAN** (`clean.diabetes_clean`): características listas para entrenar, con la partición
  asignada y trazabilidad a RAW (`raw_id`).
- **INFERENCE** (`inference.prediction_logs`): cada solicitud con su entrada, características,
  predicción, probabilidad, modelo, versión, alias, latencia y `request_id`. La vista
  `inference.v_hourly_summary` resume el tráfico por hora.

### 9.2 Idempotencia, periodicidad y fallos del DAG

- **Periodicidad:** `@hourly` (configurable con `PIPELINE_SCHEDULE`). Con `max_active_runs=1`
  nunca hay dos ejecuciones compitiendo por el mismo offset.
- **Sin duplicados:** el offset se calcula desde los lotes `COMPLETED` en la BD, no desde
  memoria. Además, `row_hash` tiene restricción `UNIQUE` con `ON CONFLICT DO NOTHING`. Si se
  reintenta la misma ejecución, se reutiliza el lote que ya cargó.
- **Atomicidad:** cada tarea escribe en una sola transacción (con un *advisory lock* en la
  carga). Si falla, hace rollback y la siguiente ejecución retoma: lo que quedó en NEW o VALID
  se procesa en la próxima ejecución.
- **Fallos visibles:** si más del 10 % de las filas del lote se rechazan, la tarea falla en
  Airflow y el reporte queda en `raw.quality_checks`. Si una partición queda sin suficientes
  positivos, también falla. Cada tarea tiene 1 reintento.

### 9.3 Transformaciones (`features.py`)

| Decisión | Justificación |
|---|---|
| Se eliminan `weight` y `payer_code` | ~97 % y ~40 % de faltantes; `payer_code` es administrativa |
| Se eliminan `examide` y `citoglipton` | Constantes: varianza cero |
| Se excluyen altas por fallecimiento u hospicio (IDs 11, 13, 14, 19, 20, 21) | Esos pacientes no pueden ser readmitidos |
| `diag_1/2/3` se agrupan en 9 categorías ICD-9 | Más de 700 códigos con muy pocos casos cada uno |
| `age` se convierte en el punto medio del rango | Preserva el orden |
| IDs de admisión y alta se agrupan | Agrupa códigos "NULL", "Not Mapped" y similares en Unknown |
| `medical_specialty`: 13 principales + Surgery + Other + Unknown | Alta cardinalidad; `?` = Unknown |
| Nulos categóricos → `Unknown`; `max_glu_serum`/`A1Cresult` faltantes → `None` | "None" significa "no medido", una categoría con significado clínico |
| Nuevas variables: `service_utilization`, `num_meds_changed`, `num_meds_active` | Resumen de visitas previas y de los 23 medicamentos |
| One-hot encoding (`handle_unknown="ignore"`, `min_frequency=20`) dentro del Pipeline | Robusto a categorías nuevas en producción |

`features.py` está en `airflow/dags/diabetes_pipeline/` y en `api/`, con exactamente el mismo
contenido. Así el modelo recibe en producción las mismas columnas con las que se entrenó, lo que
evita el *training/serving skew*. La prueba `tests/test_features.py` falla si las dos copias
dejan de ser idénticas, así que si modifica una, copie el cambio a la otra.

### 9.4 Particiones

La partición se asigna con un hash de `patient_nbr`: 70 % train, 15 % val y 15 % test. El
mismo paciente siempre queda en la misma partición aunque llegue en lotes distintos, así que
no hay fuga entre train y test, y la partición es reproducible entre ejecuciones.

### 9.5 Métrica principal y selección de modelos

La métrica principal es **F1 de la clase positiva (`<30`)**. La clase positiva es ~11 % del
total, así que un modelo que siempre responde "NO" tendría ~89 % de accuracy y sería inútil.
F1 equilibra recall (detectar pacientes en riesgo) y precisión (no saturar al equipo clínico
con falsas alarmas). También se registran ROC-AUC y PR-AUC como apoyo, porque no dependen del
umbral. Se puede cambiar con `PRIMARY_METRIC`.

Por ejecución se entrenan 3 candidatos en paralelo: regresión logística, random forest y
HistGradientBoosting, todos con `class_weight` balanceado. El mejor en validación se registra
como `@challenger`. Luego se re-evalúa el `@champion` sobre **el mismo conjunto de prueba
vigente**, y solo se promueve el challenger si lo supera. El campeón anterior queda como
`@previous-champion` para poder hacer rollback.

### 9.6 Carga del modelo en la API

La API resuelve `models:/diabetes-readmission@champion` al iniciar y carga esa versión una sola
vez. Un hilo consulta cada 60 s a qué versión apunta el alias y, si cambió, descarga la nueva y
la reemplaza de forma atómica, sin reiniciar el pod. `POST /reload-model` fuerza la consulta.
Si MLflow cae, la API sigue sirviendo con el último modelo cargado. Nunca se descarga el modelo
por solicitud y no hay versiones ni rutas escritas en el código. El registro en la BD se hace
como *BackgroundTask*, así que no suma latencia a la respuesta.

### 9.7 Versiones fijadas (problemas encontrados en la validación)

| Hallazgo | Solución |
|---|---|
| `mlflow` 2.x exige `packaging<25`, y Airflow 3.1.8 exige `packaging>=25`: no se pueden instalar juntos | MLflow **3.4.0** en todas las imágenes |
| MLflow 3.4 falla al arrancar con SQLAlchemy 2.1 (`FallbackAsyncAdaptedQueuePool`) | `sqlalchemy==2.0.36` en la imagen de MLflow |
| La imagen `apache/airflow:3.1.8` ya trae numpy 2.4.2, pandas 2.3.3 y scikit-learn 1.8.0 | Mismas versiones en Airflow y en la API (el pickle del modelo depende de ellas) |
| El chart de Airflow más reciente ya no corresponde a Airflow 3.1.8 | Chart fijado en 1.20.0 (`AIRFLOW_CHART_VERSION`) |
| `build_features` tardaba 32 ms por fila en la API | Reescrito con numpy: 9,5 ms, con salida idéntica verificada sobre 40.025 filas |

## 10. Validación realizada

Antes de entregar la estructura se ejecutó una prueba de punta a punta (fuera de Kubernetes)
con PostgreSQL 16, el servidor MLflow 3.4.0 levantado con el mismo comando del Deployment,
Airflow 3.1.8 real (`airflow dags test`) y un dataset sintético de 40.025 filas con el esquema
del original:

- 4 ejecuciones del DAG: 3 lotes (15.000 + 15.000 + 10.025) y una cuarta que se detuvo en el
  short-circuit al no quedar datos.
- RAW: 60 filas rechazadas, 25 duplicadas y 8.880 excluidas, cada una con su motivo. Ningún
  paciente quedó en más de una partición.
- MLflow: 9 runs con métricas y artefactos; versiones v1 → v2 → v3 promovidas por comparación
  contra el campeón, con alias y tags. Una quinta ejecución con `force_retrain` generó v4, que
  empató con el campeón y fue rechazada (rama `keep_current_champion`), como corresponde.
- API: el hot-swap funcionó (mover `@champion` de v3 a v1 cambió el modelo servido sin
  reiniciar), los 422 se devuelven correctamente y hubo 0 fallos al registrar en BD.
- Streamlit: los tres escenarios de ejemplo funcionaron.
- Locust: 0 fallos.

Las métricas de calidad de esa prueba no son representativas, porque los datos eran
sintéticos. Los resultados reales saldrán al ejecutar el DAG con el dataset del curso.

Los manifiestos de `k8s/` se validaron con kubeconform (43 recursos válidos en Kubernetes 1.30)
y los values de Helm contra el esquema oficial del chart 1.20.0.

## 11. Problemas comunes

| Síntoma | Causa probable / solución |
|---|---|
| PVC en `Pending` | No hay StorageClass por defecto (clúster RKE2): ver §3.2 |
| `ImagePullBackOff` en imágenes propias | No se reemplazó `DOCKERHUB_USER`, la imagen no se publicó o el tag no coincide |
| Cambié código y Kubernetes sigue usando la versión anterior | Construya con un tag nuevo y actualícelo en el manifiesto (con el mismo tag, el nodo reutiliza la imagen que ya tiene) |
| Pods de Airflow en `CreateContainerConfigError` | Se instaló Airflow antes de `kubectl apply -f k8s/` (faltan `pipeline-config` y `postgres-data-secret`) |
| La API responde 503 en `/predict` | Todavía no hay `@champion`: ejecute el DAG |
| `check_source_file` falla | Falta `airflow/data/Diabetes.csv` en la imagen y el pod no tiene salida a Internet |
| El DAG no aparece | `kubectl logs deploy/airflow-dag-processor`; verifique el tag de la imagen de Airflow |
| Error `invalid choice: 'api-server'` en Airflow | Chart e imagen desalineados: chart 1.20.0 ↔ Airflow 3.1.8 |
| Grafana sin datos de CPU de cAdvisor | El clúster restringe `nodes/proxy`; las métricas del proceso de la API funcionan igual |
| Cambió un `.sql` | Regenere `k8s/db-sql-configmap.yaml`, luego `kubectl delete job data-db-migrate` y `kubectl apply -f k8s/` |
| No abre `http://IP_NODO:30xxx` en el clúster `local` | Use `kubectl port-forward --address 0.0.0.0` (§6) |
