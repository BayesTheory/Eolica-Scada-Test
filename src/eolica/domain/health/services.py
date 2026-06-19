"""Serviço de domínio: converter erros de reconstrução num veredito de saúde.

Função pura. Sem I/O, sem modelo, sem framework — dá para ler a regra inteira
sem saber que existe um LSTM do outro lado.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from eolica.domain.health.value_objects import (
    AnomalyThreshold,
    HealthStatus,
    ReconstructionError,
)
from eolica.shared.errors import InsufficientDataError, InvalidValueError


@dataclass(frozen=True, slots=True)
class HealthVerdict:
    """O resultado da avaliação, com o rastro necessário para auditá-la.

    Guardar `threshold`, `exceedances` e `evaluated_windows` junto do status é o
    que permite responder "por que a turbina foi marcada como ALERTA no dia 7?"
    sem reprocessar nada.
    """

    status: HealthStatus
    exceedances: int
    """Janelas acima do limiar, incluindo picos isolados."""

    sustained_anomalies: int
    """Janelas que fazem parte de uma corrida longa o bastante para alarmar."""

    evaluated_windows: int
    threshold: AnomalyThreshold
    persistence_window: int
    previous_period_anomalies: int | None
    reason: str

    @property
    def previous_period_known(self) -> bool:
        """False quando não há informação sobre o período anterior.

        Distinto de "o período anterior teve zero anomalias". O v1 colapsava os
        dois casos num `-1` que o prompt do LLM comparava com `> 0`.
        """
        return self.previous_period_anomalies is not None


def _count_sustained(flags: Sequence[bool], persistence_window: int) -> int:
    """Soma o comprimento das corridas de `True` com pelo menos N elementos."""
    sustained = 0
    run = 0
    for flag in flags:
        if flag:
            run += 1
            continue
        if run >= persistence_window:
            sustained += run
        run = 0
    if run >= persistence_window:
        sustained += run
    return sustained


def evaluate_health(
    *,
    errors: Sequence[ReconstructionError],
    threshold: AnomalyThreshold,
    persistence_window: int,
    previous_period_anomalies: int | None = None,
) -> HealthVerdict:
    """Decide o estado de saúde da turbina para um período.

    A regra tem duas partes, e as duas estavam fora do código no v1:

    **Persistência.** Uma janela isolada acima do limiar não alarma. Num sinal
    amostrado a cada 10 minutos, um pico único é quase sempre ruído de sensor.
    Só uma corrida de `persistence_window` janelas consecutivas conta — com o
    valor 6 do `config.yaml`, isso significa uma hora de desvio contínuo.

    **Manutenção.** Anomalia sustentada hoje *e* anomalia ontem indica trabalho
    em curso na máquina, não uma falha nova a reportar. Esta regra vivia na
    instrução 4 do prompt do co-piloto, o que a tornava invisível para qualquer
    consumidor da API que não fosse o chat.

    Args:
        errors: erros de reconstrução, em ordem cronológica. A ordem importa:
            a persistência é medida sobre janelas *consecutivas*.
        threshold: o limiar a aplicar.
        persistence_window: janelas consecutivas necessárias para alarmar.
        previous_period_anomalies: anomalias sustentadas no período anterior, ou
            `None` quando não se sabe. `None` nunca conclui manutenção.
    """
    if persistence_window < 1:
        raise InvalidValueError(
            "A janela de persistência deve ser de pelo menos 1",
            persistence_window=persistence_window,
        )
    if not errors:
        raise InsufficientDataError(required=1, available=0, subject="erros de reconstrução")
    if previous_period_anomalies is not None and previous_period_anomalies < 0:
        raise InvalidValueError(
            "A contagem de anomalias do período anterior não pode ser negativa. "
            "Use None para indicar ausência de informação",
            previous_period_anomalies=previous_period_anomalies,
        )

    flags = [threshold.is_exceeded_by(error) for error in errors]
    exceedances = sum(flags)
    sustained = _count_sustained(flags, persistence_window)

    status, reason = _classify(
        exceedances=exceedances,
        sustained=sustained,
        persistence_window=persistence_window,
        previous_period_anomalies=previous_period_anomalies,
    )

    return HealthVerdict(
        status=status,
        exceedances=exceedances,
        sustained_anomalies=sustained,
        evaluated_windows=len(errors),
        threshold=threshold,
        persistence_window=persistence_window,
        previous_period_anomalies=previous_period_anomalies,
        reason=reason,
    )


def _classify(
    *,
    exceedances: int,
    sustained: int,
    persistence_window: int,
    previous_period_anomalies: int | None,
) -> tuple[HealthStatus, str]:
    if sustained == 0:
        if exceedances == 0:
            return HealthStatus.OK, "Nenhuma janela acima do limiar."
        return (
            HealthStatus.OK,
            f"{exceedances} janela(s) acima do limiar, mas nenhuma corrida atingiu as "
            f"{persistence_window} janelas consecutivas exigidas: tratado como ruído.",
        )

    if previous_period_anomalies is not None and previous_period_anomalies > 0:
        return (
            HealthStatus.UNDER_MAINTENANCE,
            f"Anomalia sustentada por {sustained} janela(s) hoje e "
            f"{previous_period_anomalies} no período anterior: indica intervenção em curso.",
        )

    return (
        HealthStatus.ALERT,
        f"Anomalia sustentada por {sustained} janela(s) consecutivas "
        f"(mínimo {persistence_window}), sem ocorrência no período anterior.",
    )
