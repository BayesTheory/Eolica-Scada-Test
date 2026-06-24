"""Value objects do subdomínio `forecasting`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from eolica.domain.turbine import PowerKw, TurbineSpec
from eolica.shared.errors import InvalidValueError


@dataclass(frozen=True, slots=True)
class Horizon:
    """Quão longe no futuro a previsão alcança.

    Existe como tipo próprio porque "prever o próximo passo" é ambíguo: um passo
    de 10 minutos e um de uma hora são produtos diferentes para o operador. O v1
    dizia apenas `predict_next_step` e o consumidor tinha que adivinhar.
    """

    steps: int
    step: timedelta

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise InvalidValueError("O horizonte deve ter pelo menos 1 passo", steps=self.steps)
        if self.step <= timedelta(0):
            raise InvalidValueError("A duração do passo deve ser positiva", step=str(self.step))

    @property
    def duration(self) -> timedelta:
        return self.steps * self.step


@dataclass(frozen=True, slots=True)
class PowerForecast:
    """Uma previsão de potência, com tudo que é preciso para auditá-la.

    `model_version` não é opcional de propósito. Uma previsão sem a identidade
    do modelo que a produziu é impossível de investigar depois — e investigar
    previsão ruim é metade do trabalho de manter um modelo em produção.
    """

    power: PowerKw
    issued_at: datetime
    horizon: Horizon
    model_version: str

    @property
    def target_time(self) -> datetime:
        """O instante ao qual a previsão se refere."""
        return self.issued_at + self.horizon.duration

    def for_display(self) -> float:
        """Potência prevista como deve ser mostrada: nunca negativa."""
        return self.power.for_display()

    def exceeds_rated(self, spec: TurbineSpec) -> bool:
        """True se a previsão passa do teto físico da máquina.

        Previsão acima da potência nominal não é vento excepcional: é o modelo
        extrapolando para fora do domínio em que foi treinado. Vale monitorar,
        não corrigir silenciosamente.
        """
        return self.power.kw > spec.rated_power_kw
