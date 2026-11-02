"""Métricas Prometheus de la API y middleware de instrumentación."""
import time

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75,
                   1.0, 2.5, 5.0, 10.0)

HTTP_REQUESTS = Counter(
    "inference_http_requests_total", "Solicitudes HTTP recibidas",
    ["method", "handler", "status"])
HTTP_LATENCY = Histogram(
    "inference_http_request_duration_seconds", "Latencia de las solicitudes HTTP",
    ["method", "handler"], buckets=LATENCY_BUCKETS)
HTTP_IN_PROGRESS = Gauge(
    "inference_http_requests_in_progress", "Solicitudes en curso")
MODEL_LATENCY = Histogram(
    "inference_model_predict_seconds", "Tiempo de predict_proba del modelo",
    buckets=LATENCY_BUCKETS)
PREDICTIONS = Counter(
    "inference_predictions_total", "Predicciones generadas",
    ["model_name", "model_version", "predicted_label"])
PREDICTION_ERRORS = Counter(
    "inference_prediction_errors_total", "Errores al predecir", ["reason"])
DB_LOG_FAILURES = Counter(
    "inference_db_log_failures_total", "Fallos al registrar inferencias en la BD")
MODEL_INFO = Gauge(
    "inference_model_info", "Modelo cargado (valor 1)",
    ["model_name", "model_version", "model_alias", "run_id"])
MODEL_LOADED = Gauge("inference_model_loaded", "1 si hay un modelo cargado")
MODEL_RELOADS = Counter("inference_model_reloads_total", "Recargas de modelo", ["result"])
PREDICTED_PROBABILITY = Histogram(
    "inference_predicted_probability", "Distribución del score (monitoreo de deriva)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Mide todas las rutas usando la plantilla de la ruta (evita alta cardinalidad)."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        status = "500"
        HTTP_IN_PROGRESS.inc()
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            HTTP_IN_PROGRESS.dec()
            route = request.scope.get("route")
            label = getattr(route, "path", None) or "unmatched"
            HTTP_REQUESTS.labels(request.method, label, status).inc()
            HTTP_LATENCY.labels(request.method, label).observe(time.perf_counter() - start)
