"""Métricas de avaliação de modelo — em Python puro, como todo o domínio."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from eolica.shared.errors import InsufficientDataError, InvalidValueError


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """RMSE, MAE e R² de uma previsão contínua."""

    rmse: float
    mae: float
    r2: float

    @classmethod
    def of(cls, *, actual: Sequence[float], predicted: Sequence[float]) -> RegressionMetrics:
        if len(actual) != len(predicted):
            raise InvalidValueError(
                "Observado e previsto devem ter o mesmo tamanho",
                actual=len(actual),
                predicted=len(predicted),
            )
        if not actual:
            raise InsufficientDataError(required=1, available=0, subject="observações")

        size = len(actual)
        errors = [a - p for a, p in zip(actual, predicted, strict=True)]
        mean_actual = sum(actual) / size

        squared_error = sum(error**2 for error in errors)
        total_variance = sum((value - mean_actual) ** 2 for value in actual)

        # Alvo constante: a variância total é zero e o R² fica indefinido. Se o
        # modelo também acertou, a previsão é perfeita; senão, é o pior caso.
        if total_variance == 0:
            r2 = 1.0 if squared_error == 0 else float("-inf")
        else:
            r2 = 1.0 - squared_error / total_variance

        return cls(
            rmse=math.sqrt(squared_error / size),
            mae=sum(abs(error) for error in errors) / size,
            r2=r2,
        )


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Matriz de confusão de um detector binário.

    `false_alarm_rate` é a métrica que mais importa aqui e a que o v1 nunca
    mediu: um detector que alarma o tempo todo é desligado pelo operador na
    primeira semana, e aí a recall real vira zero.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @classmethod
    def of(cls, *, predicted: Sequence[bool], actual: Sequence[bool]) -> DetectionMetrics:
        if len(predicted) != len(actual):
            raise InvalidValueError(
                "Previsto e observado devem ter o mesmo tamanho",
                predicted=len(predicted),
                actual=len(actual),
            )
        if not predicted:
            raise InsufficientDataError(required=1, available=0, subject="observações")

        pairs = list(zip(predicted, actual, strict=True))
        return cls(
            true_positives=sum(1 for p, a in pairs if p and a),
            false_positives=sum(1 for p, a in pairs if p and not a),
            false_negatives=sum(1 for p, a in pairs if not p and a),
            true_negatives=sum(1 for p, a in pairs if not p and not a),
        )

    @property
    def precision(self) -> float:
        """Dos alarmes emitidos, quantos eram reais."""
        alarms = self.true_positives + self.false_positives
        return 0.0 if alarms == 0 else self.true_positives / alarms

    @property
    def recall(self) -> float:
        """Dos eventos reais, quantos foram detectados."""
        events = self.true_positives + self.false_negatives
        return 0.0 if events == 0 else self.true_positives / events

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 0.0 if denominator == 0 else 2 * self.precision * self.recall / denominator

    @property
    def false_alarm_rate(self) -> float:
        """Fração dos períodos saudáveis em que houve alarme."""
        healthy = self.false_positives + self.true_negatives
        return 0.0 if healthy == 0 else self.false_positives / healthy


@dataclass(frozen=True, slots=True)
class PromotionVerdict:
    """A decisão de promover — ou não — um modelo desafiante."""

    approved: bool
    improvement: float
    """Redução relativa de RMSE sobre o baseline. Negativa se piorou."""

    challenger: RegressionMetrics
    baseline: RegressionMetrics
    min_improvement: float
    reason: str
