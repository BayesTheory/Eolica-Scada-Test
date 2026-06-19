"""Value objects do subdomínio `health`."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from eolica.shared.errors import InsufficientDataError, InvalidValueError


@dataclass(frozen=True, slots=True, order=True)
class ReconstructionError:
    """Erro de reconstrução de uma janela pelo autoencoder (MSE).

    Quanto maior, mais a janela destoa do que o modelo aprendeu como "operação
    normal". É uma média de quadrados: nunca negativo.
    """

    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise InvalidValueError("Erro de reconstrução deve ser finito", value=repr(self.value))
        if self.value < 0:
            raise InvalidValueError("Erro de reconstrução não pode ser negativo", value=self.value)


class ThresholdMethod(StrEnum):
    """Como o limiar de anomalia foi derivado.

    Guardar isto junto do valor não é burocracia: o v1 tinha
    `threshold_std: 3.0` no `config.yaml` e `np.percentile(erros, 99.5)` no
    código. Os dois números descrevem limiares diferentes, ninguém sabia qual
    estava valendo, e o do config não valia nada — não era lido por lugar nenhum.
    """

    PERCENTILE = "percentile"
    STD_DEVIATIONS = "std_deviations"


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Percentil com interpolação linear, compatível com `numpy.percentile`.

    Reimplementado em Python puro para manter o domínio sem dependências. A
    compatibilidade numérica com numpy é testada: o limiar do v1 era calculado
    com numpy no serving, e uma divergência silenciosa aqui mudaria o
    comportamento do detector sem aparecer em nenhum diff de lógica.
    """
    ordered = sorted(values)
    size = len(ordered)
    if size == 1:
        return ordered[0]

    rank = (size - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])


@dataclass(frozen=True, slots=True)
class AnomalyThreshold:
    """O limiar acima do qual uma janela é considerada anômala.

    Imutável e autodescritivo: carrega o método e o parâmetro que o produziram,
    para que uma decisão de alerta possa ser reproduzida meses depois.
    """

    value: float
    method: ThresholdMethod
    parameter: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise InvalidValueError("Limiar deve ser finito", value=repr(self.value))

    @classmethod
    def from_percentile(
        cls, errors: Sequence[ReconstructionError], *, percentile: float
    ) -> AnomalyThreshold:
        """Limiar como percentil dos erros observados em operação normal."""
        if not 0.0 <= percentile <= 100.0:
            raise InvalidValueError("O percentil deve estar em [0, 100]", percentile=percentile)
        if not errors:
            raise InsufficientDataError(required=1, available=0, subject="erros de referência")

        return cls(
            value=_percentile([e.value for e in errors], percentile),
            method=ThresholdMethod.PERCENTILE,
            parameter=percentile,
        )

    @classmethod
    def from_std_deviations(
        cls, errors: Sequence[ReconstructionError], *, n_std: float
    ) -> AnomalyThreshold:
        """Limiar como média + n·σ, com σ populacional (equivale a `np.std`).

        Este é o método que o `config.yaml` do v1 declarava (`threshold_std: 3.0`)
        e que o código nunca chegou a usar. Agora é uma opção de verdade.
        """
        if n_std < 0:
            raise InvalidValueError("O número de desvios padrão deve ser positivo", n_std=n_std)
        if not errors:
            raise InsufficientDataError(required=1, available=0, subject="erros de referência")

        values = [e.value for e in errors]
        spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        return cls(
            value=statistics.fmean(values) + n_std * spread,
            method=ThresholdMethod.STD_DEVIATIONS,
            parameter=n_std,
        )

    def is_exceeded_by(self, error: ReconstructionError) -> bool:
        """Comparação estrita, como no detector original."""
        return error.value > self.value


class HealthStatus(StrEnum):
    """Veredito de saúde da turbina para um período."""

    OK = "OK"
    ALERT = "ALERTA"
    UNDER_MAINTENANCE = "EM_MANUTENCAO"
