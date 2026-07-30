"""Subdomínio `evaluation`: quanto vale um modelo, e se ele merece produção.

Contexto de suporte compartilhado por saúde e previsão. A decisão de promover é
regra de negócio — a organização define o que considera melhoria suficiente — e
não um detalhe do script de treino.
"""

from eolica.domain.evaluation.services import compare_against_baseline
from eolica.domain.evaluation.value_objects import (
    DetectionMetrics,
    PromotionVerdict,
    RegressionMetrics,
)

__all__ = [
    "DetectionMetrics",
    "PromotionVerdict",
    "RegressionMetrics",
    "compare_against_baseline",
]
