"""Backtest da janela de persistência.

Responde a uma pergunta que o v1 não tinha como responder: **quanto a janela de
persistência realmente vale?**

O `config.yaml` do v1 declarava `persistence_window: 6` e nenhuma linha o lia,
então o detector alarmava em qualquer janela isolada acima do limiar. Reintroduzir
o parâmetro sem medir seria trocar um palpite não implementado por um palpite
implementado.

Este módulo varre o histórico inteiro com vários valores de janela e reporta,
para cada um, quantos episódios de alarme seriam abertos e qual a taxa de alarme
falso — usando os períodos de falha reportados pelo próprio SCADA como
referência.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from eolica.domain.evaluation import DetectionMetrics
from eolica.domain.health.value_objects import AnomalyThreshold, ReconstructionError
from eolica.shared.errors import InsufficientDataError, InvalidValueError


@dataclass(frozen=True, slots=True)
class AlarmEpisode:
    """Uma corrida contígua de janelas acima do limiar, longa o bastante para
    alarmar. É a unidade que chega ao operador — não a janela individual."""

    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length - 1


@dataclass(frozen=True, slots=True)
class PersistenceOutcome:
    """O que aconteceria com um dado valor de janela de persistência."""

    persistence_window: int
    episodes: int
    alarming_windows: int
    metrics: DetectionMetrics

    @property
    def false_alarm_rate(self) -> float:
        return self.metrics.false_alarm_rate

    @property
    def recall(self) -> float:
        return self.metrics.recall


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """Comparação entre valores de janela de persistência."""

    outcomes: tuple[PersistenceOutcome, ...]
    evaluated_windows: int
    threshold: AnomalyThreshold

    def outcome_for(self, persistence_window: int) -> PersistenceOutcome:
        for outcome in self.outcomes:
            if outcome.persistence_window == persistence_window:
                return outcome
        raise InvalidValueError(
            "Janela de persistência não avaliada neste backtest",
            requested=persistence_window,
            evaluated=[o.persistence_window for o in self.outcomes],
        )

    def false_alarms_avoided(self, *, baseline: int, candidate: int) -> int:
        """Alarmes falsos que `candidate` evita em relação a `baseline`.

        Com `baseline=1` (o comportamento efetivo do v1, que alarmava em
        qualquer pico isolado), este número é o argumento a favor da janela.
        """
        return (
            self.outcome_for(baseline).metrics.false_positives
            - self.outcome_for(candidate).metrics.false_positives
        )

    def detections_lost(self, *, baseline: int, candidate: int) -> int:
        """Eventos reais que `candidate` deixa de detectar. É o custo da janela.

        Reportado ao lado do ganho de propósito: uma janela grande demais suprime
        ruído e evento real junto, e a escolha do valor é um trade-off explícito,
        não uma otimização de uma métrica só.
        """
        return (
            self.outcome_for(candidate).metrics.false_negatives
            - self.outcome_for(baseline).metrics.false_negatives
        )


def find_alarm_episodes(flags: Sequence[bool], *, persistence_window: int) -> list[AlarmEpisode]:
    """Corridas contíguas de `True` com pelo menos `persistence_window` itens."""
    if persistence_window < 1:
        raise InvalidValueError(
            "A janela de persistência deve ser de pelo menos 1",
            persistence_window=persistence_window,
        )

    episodes: list[AlarmEpisode] = []
    position = 0
    for flag, group in itertools.groupby(flags):
        length = len(list(group))
        if flag and length >= persistence_window:
            episodes.append(AlarmEpisode(start=position, length=length))
        position += length
    return episodes


def backtest_persistence(
    *,
    errors: Sequence[ReconstructionError],
    threshold: AnomalyThreshold,
    is_real_event: Sequence[bool],
    persistence_windows: Sequence[int],
) -> BacktestReport:
    """Avalia várias janelas de persistência sobre o mesmo histórico.

    Args:
        errors: erros de reconstrução em ordem cronológica.
        threshold: o limiar em vigor.
        is_real_event: para cada janela, se o SCADA reportou falha nela. É a
            referência disponível — não é rótulo de especialista, e vale
            interpretá-la como proxy: o status de falha aparece *durante* a
            falha, não antes, então a recall aqui subestima a capacidade de
            alerta precoce.
        persistence_windows: os valores a comparar.
    """
    if not errors:
        raise InsufficientDataError(required=1, available=0, subject="erros de reconstrução")
    if len(errors) != len(is_real_event):
        raise InvalidValueError(
            "Erros e rótulos devem ter o mesmo tamanho",
            errors=len(errors),
            labels=len(is_real_event),
        )
    if not persistence_windows:
        raise InsufficientDataError(
            required=1, available=0, subject="janelas de persistência a comparar"
        )

    flags = [threshold.is_exceeded_by(error) for error in errors]

    outcomes: list[PersistenceOutcome] = []
    for window in sorted(set(persistence_windows)):
        episodes = find_alarm_episodes(flags, persistence_window=window)

        alarming = [False] * len(flags)
        for episode in episodes:
            for index in range(episode.start, episode.end + 1):
                alarming[index] = True

        outcomes.append(
            PersistenceOutcome(
                persistence_window=window,
                episodes=len(episodes),
                alarming_windows=sum(alarming),
                metrics=DetectionMetrics.of(predicted=alarming, actual=list(is_real_event)),
            )
        )

    return BacktestReport(
        outcomes=tuple(outcomes), evaluated_windows=len(errors), threshold=threshold
    )
