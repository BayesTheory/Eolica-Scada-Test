"""Caso de uso: quanto a janela de persistência vale sobre o histórico real.

Alimenta o backtest de domínio com a telemetria inteira, respeitando as
descontinuidades — um segmento por vez, como o detector faz em produção.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from eolica.domain.health import AnomalyThreshold, ReconstructionError
from eolica.domain.health.backtest import BacktestReport, backtest_persistence
from eolica.domain.turbine import OperatingStatus, ReadingWindow

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eolica.application.ports import ScadaRepository
    from eolica.domain.health import ReconstructionModel

DEFAULT_PERSISTENCE_WINDOWS = (1, 2, 3, 6, 12)


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    """O relatório, mais o contexto de quanto do histórico foi coberto."""

    report: BacktestReport
    total_readings: int
    analysed_segments: int
    real_event_windows: int


class BacktestDetector:
    """Varre o histórico e compara valores de janela de persistência."""

    def __init__(
        self,
        *,
        readings: ScadaRepository,
        health_model: ReconstructionModel,
        threshold: AnomalyThreshold,
        sampling_interval: timedelta,
        persistence_windows: Sequence[int] = DEFAULT_PERSISTENCE_WINDOWS,
    ) -> None:
        self._readings = readings
        self._health_model = health_model
        self._threshold = threshold
        self._sampling_interval = sampling_interval
        self._persistence_windows = tuple(persistence_windows)

    def execute(self) -> BacktestSummary:
        start, end = self._readings.available_range()
        everything = self._readings.readings_between(start, end)

        segments = ReadingWindow.split_on_gaps(
            everything,
            expected_interval=self._sampling_interval,
            min_length=self._health_model.window_size,
        )

        errors: list[ReconstructionError] = []
        labels: list[bool] = []
        for segment in segments:
            errors.extend(self._health_model.reconstruction_errors(segment))
            labels.extend(self._segment_labels(segment))

        return BacktestSummary(
            report=backtest_persistence(
                errors=errors,
                threshold=self._threshold,
                is_real_event=labels,
                persistence_windows=self._persistence_windows,
            ),
            total_readings=len(everything),
            analysed_segments=len(segments),
            real_event_windows=sum(labels),
        )

    def _segment_labels(self, segment: ReadingWindow) -> list[bool]:
        """Uma sub-janela conta como evento real se contém leitura em falha.

        A referência é o próprio código de status do SCADA (13). Não é rótulo de
        especialista, e a limitação vale ser dita: o status de falha aparece
        *durante* a falha, não antes dela. Ou seja, a recall medida aqui
        subestima a capacidade de alerta precoce — que é justamente o que um
        detector por reconstrução deveria oferecer.

        Medir com a referência imperfeita disponível e declarar a imperfeição é
        melhor que não medir.
        """
        size = self._health_model.window_size
        faults = [reading.status is OperatingStatus.FAULT for reading in segment.readings]
        count = len(faults) - size + 1
        if count < 1:
            return []
        return [any(faults[start : start + size]) for start in range(count)]
