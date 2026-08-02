"""Caso de uso: cobertura de telemetria dia a dia, num período.

Existe porque a fragmentação da série é a informação mais importante que o v1
escondia — e escondê-la num campo do relatório de um único dia não basta. Ver
os buracos ao longo de semanas é o que revela que 2022-01-20 não é uma exceção.

Deliberadamente **não** roda modelo nenhum: é uma varredura de índice, então
responde em milissegundos mesmo cobrindo o acervo inteiro.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

from eolica.domain.turbine import OperatingStatus, ReadingWindow
from eolica.shared.errors import InvalidValueError

if TYPE_CHECKING:
    from eolica.application.ports import ScadaRepository

MAX_RANGE_DAYS = 400


@dataclass(frozen=True, slots=True)
class DayCoverage:
    """Cobertura de um dia civil."""

    day: date
    readings: int
    expected_readings: int
    segments: int
    longest_segment: int
    faulted_readings: int

    @property
    def completeness(self) -> float:
        if self.expected_readings == 0:
            return 0.0
        return min(1.0, self.readings / self.expected_readings)

    @property
    def is_fragmented(self) -> bool:
        return self.segments > 1

    @property
    def is_absent(self) -> bool:
        """Dia sem nenhuma leitura — buraco de dia inteiro no acervo.

        Distinto de um dia fragmentado: aqui não há o que analisar.
        """
        return self.readings == 0


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    days: tuple[DayCoverage, ...]
    start: date
    end: date

    @property
    def mean_completeness(self) -> float:
        if not self.days:
            return 0.0
        return sum(day.completeness for day in self.days) / len(self.days)

    @property
    def fragmented_days(self) -> int:
        return sum(1 for day in self.days if day.is_fragmented)

    @property
    def absent_days(self) -> int:
        return sum(1 for day in self.days if day.is_absent)


class SummariseCoverage:
    """Varre o índice e reporta cobertura por dia."""

    def __init__(self, *, readings: ScadaRepository, sampling_interval: timedelta) -> None:
        self._readings = readings
        self._sampling_interval = sampling_interval

    def execute(self, *, start: date | None = None, end: date | None = None) -> CoverageSummary:
        """Cobertura de cada dia do período. Sem argumentos, cobre o acervo todo."""
        archive_start, archive_end = self._readings.available_range()
        first = start or archive_start.date()
        last = end or archive_end.date()

        if last < first:
            raise InvalidValueError(
                "A data final não pode ser anterior à inicial", start=str(first), end=str(last)
            )
        span = (last - first).days + 1
        if span > MAX_RANGE_DAYS:
            raise InvalidValueError(
                f"O período pedido excede o máximo de {MAX_RANGE_DAYS} dias",
                requested_days=span,
                maximum=MAX_RANGE_DAYS,
            )

        expected = int(timedelta(days=1) / self._sampling_interval)
        tzinfo = archive_start.tzinfo

        days: list[DayCoverage] = []
        for offset in range(span):
            current = first + timedelta(days=offset)
            readings = self._readings.readings_between(
                datetime.combine(current, time.min, tzinfo=tzinfo),
                datetime.combine(current, time.max, tzinfo=tzinfo),
            )
            segments = (
                ReadingWindow.split_on_gaps(readings, expected_interval=self._sampling_interval)
                if readings
                else []
            )
            days.append(
                DayCoverage(
                    day=current,
                    readings=len(readings),
                    expected_readings=expected,
                    segments=len(segments),
                    longest_segment=max((len(s) for s in segments), default=0),
                    faulted_readings=sum(1 for r in readings if r.status is OperatingStatus.FAULT),
                )
            )

        return CoverageSummary(days=tuple(days), start=first, end=last)
