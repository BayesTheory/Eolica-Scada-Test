"""Subdomínio `monitoring`: o modelo ainda descreve o mundo que vê?

Contexto de suporte, ausente no v1. Sem ele um autoencoder treinado em 2022
segue emitindo erros de reconstrução em 2024 com toda a confiança do mundo,
e ninguém tem como saber que a distribuição de entrada mudou embaixo dele.
"""

from eolica.domain.monitoring.services import kolmogorov_smirnov, population_stability_index
from eolica.domain.monitoring.value_objects import (
    DriftMethod,
    DriftReport,
    DriftScore,
    DriftSeverity,
)

__all__ = [
    "DriftMethod",
    "DriftReport",
    "DriftScore",
    "DriftSeverity",
    "kolmogorov_smirnov",
    "population_stability_index",
]
