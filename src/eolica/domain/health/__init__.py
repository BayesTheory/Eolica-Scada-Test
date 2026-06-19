"""Subdomínio `health`: detecção de anomalia por reconstrução.

Núcleo do produto. A pergunta que este contexto responde é "esta turbina está
se comportando como sempre se comportou quando estava saudável?" — e não
"a turbina vai falhar", que seria outro modelo e outro contexto.
"""

from eolica.domain.health.ports import ReconstructionModel
from eolica.domain.health.services import HealthVerdict, evaluate_health
from eolica.domain.health.value_objects import (
    AnomalyThreshold,
    HealthStatus,
    ReconstructionError,
    ThresholdMethod,
)

__all__ = [
    "AnomalyThreshold",
    "HealthStatus",
    "HealthVerdict",
    "ReconstructionError",
    "ReconstructionModel",
    "ThresholdMethod",
    "evaluate_health",
]
