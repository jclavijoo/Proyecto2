# Análisis exploratorio (EDA)

El enunciado evalúa el análisis completo de los datos. Cree aquí `01_eda.ipynb` con lo siguiente:

1. Distribución de `readmitted` (NO / >30 / <30) y desbalance de la clase `<30`.
2. Porcentaje de `?` por columna: justifica eliminar `weight` y `payer_code` y el manejo de
   `medical_specialty`.
3. Columnas constantes (`examide`, `citoglipton`).
4. Frecuencia de `discharge_disposition_id`: altas por fallecimiento u hospicio (11, 13, 14, 19, 20, 21).
5. Cardinalidad de `diag_1/2/3` antes y después de agrupar con `icd9_group`.
6. Pacientes con varios encuentros (`patient_nbr` repetido): justifica la partición agrupada.
7. Relación de `number_inpatient`, `number_emergency` y `time_in_hospital` con la tasa de `<30`.

Copie `api/features.py` junto al notebook e importe `from features import build_features` para que el
notebook use exactamente las mismas transformaciones que el pipeline.
