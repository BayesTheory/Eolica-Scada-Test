"""Portas da camada de aplicação.

São `Protocol`s: a aplicação diz o que precisa, a infraestrutura fornece, e
nenhuma das duas importa a outra. É o que permite testar todo caso de uso sem
CSV, sem MLflow e sem rede — os fakes em `tests/fakes.py` satisfazem estes
contratos em memória.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date, datetime

    from eolica.domain.turbine import TurbineReading


@runtime_checkable
class ScadaRepository(Protocol):
    """Acesso à telemetria histórica da turbina."""

    def readings_for_day(self, day: date) -> Sequence[TurbineReading]:
        """Leituras de um dia civil.

        Levanta `NotFoundError` se o dia não existe no acervo — **não** devolve
        lista vazia. A distinção importa: vazio é "o dia existe e não teve
        medição", ausente é "esse dia não está no dataset". O v1 fazia
        `df.loc[data_string]`, que levanta `KeyError` para data ausente; o
        `if df_dia.empty` logo abaixo nunca era alcançado e o handler devolvia
        500 para o que era um 404.
        """
        ...

    def readings_before(self, moment: datetime, *, limit: int) -> Sequence[TurbineReading]:
        """As `limit` leituras imediatamente anteriores a `moment`, em ordem."""
        ...

    def available_range(self) -> tuple[datetime, datetime]:
        """Primeiro e último instante disponíveis no acervo."""
        ...

    def readings_between(self, start: datetime, end: datetime) -> Sequence[TurbineReading]:
        """Leituras no intervalo fechado `[start, end]`."""
        ...


@runtime_checkable
class Clock(Protocol):
    """O relógio, como dependência explícita.

    `datetime.now()` espalhado pelo código torna impossível testar qualquer
    coisa sensível a tempo sem monkeypatch.
    """

    def now(self) -> datetime: ...


@runtime_checkable
class MetricsRecorder(Protocol):
    """Coleta de métricas operacionais, sem acoplar a aplicação ao Prometheus."""

    def record_inference(self, *, model: str, duration_seconds: float, outcome: str) -> None: ...

    def record_health_verdict(self, *, status: str) -> None: ...

    def record_drift(self, *, feature: str, score: float) -> None: ...
