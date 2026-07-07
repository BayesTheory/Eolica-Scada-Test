"""Métricas Prometheus.

O que se mede aqui é escolhido para responder às perguntas que um operador faz
às 3 da manhã, e não para encher dashboard:

- a API está lenta? (`eolica_request_duration_seconds`)
- o detector está alarmando mais que o normal? (`eolica_health_verdict_total`)
- o dado de entrada mudou? (`eolica_feature_drift_psi`)
- quanto de cada dia estamos realmente conseguindo analisar?
  (`eolica_report_coverage_ratio`) — a métrica que teria mostrado, no v1, que
  dias fragmentados estavam sendo analisados com janelas furadas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client import generate_latest as _generate_latest

if TYPE_CHECKING:
    from fastapi import FastAPI

REGISTRY = CollectorRegistry(auto_describe=True)

REQUEST_DURATION = Histogram(
    "eolica_request_duration_seconds",
    "Duração das requisições HTTP",
    labelnames=("method", "path", "status"),
    registry=REGISTRY,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

HEALTH_VERDICTS = Counter(
    "eolica_health_verdict_total",
    "Vereditos de saúde emitidos, por status",
    labelnames=("status",),
    registry=REGISTRY,
)

INFERENCE_DURATION = Histogram(
    "eolica_inference_duration_seconds",
    "Duração da inferência, por modelo",
    labelnames=("model", "outcome"),
    registry=REGISTRY,
)

FEATURE_DRIFT = Gauge(
    "eolica_feature_drift_psi",
    "PSI da feature entre referência e período recente",
    labelnames=("feature",),
    registry=REGISTRY,
)

REPORT_COVERAGE = Histogram(
    "eolica_report_coverage_ratio",
    "Fração do dia efetivamente analisada",
    registry=REGISTRY,
    buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
)

ANOMALY_THRESHOLD = Gauge(
    "eolica_anomaly_threshold",
    "Limiar de anomalia em vigor",
    registry=REGISTRY,
)


class PrometheusMetrics:
    """Adaptador de `application.ports.MetricsRecorder`."""

    def record_inference(self, *, model: str, duration_seconds: float, outcome: str) -> None:
        INFERENCE_DURATION.labels(model=model, outcome=outcome).observe(duration_seconds)

    def record_health_verdict(self, *, status: str) -> None:
        HEALTH_VERDICTS.labels(status=status).inc()

    def record_drift(self, *, feature: str, score: float) -> None:
        FEATURE_DRIFT.labels(feature=feature).set(score)

    def record_coverage(self, ratio: float) -> None:
        REPORT_COVERAGE.observe(ratio)

    def record_threshold(self, value: float) -> None:
        ANOMALY_THRESHOLD.set(value)


def mount_metrics(app: FastAPI) -> None:
    """Expõe `/metrics` no formato de exposição do Prometheus."""
    from fastapi import Response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=_generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
