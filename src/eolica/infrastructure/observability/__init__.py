"""Observabilidade: logging estruturado e métricas Prometheus."""

from eolica.infrastructure.observability.logging import configure_logging
from eolica.infrastructure.observability.metrics import (
    PrometheusMetrics,
    mount_metrics,
)

__all__ = ["PrometheusMetrics", "configure_logging", "mount_metrics"]
