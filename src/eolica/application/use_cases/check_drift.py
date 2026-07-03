"""Caso de uso: o modelo ainda vê o mundo em que foi treinado?

Compara a distribuição das features num período de referência (o começo do
acervo, que é aproximadamente o que o modelo viu no treino) com a de um período
recente. Não existia no v1.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from eolica.domain.monitoring import DriftReport, population_stability_index
from eolica.domain.turbine import FEATURE_NAMES
from eolica.shared.errors import InsufficientDataError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eolica.application.ports import ScadaRepository


class CheckDrift:
    """Calcula PSI por feature entre referência e período recente."""

    def __init__(
        self,
        *,
        readings: ScadaRepository,
        features: Sequence[str] = FEATURE_NAMES,
        bins: int = 10,
        window_days: int = 30,
    ) -> None:
        self._readings = readings
        self._features = tuple(features)
        self._bins = bins
        self._window_days = window_days

    def execute(self) -> DriftReport:
        """Compara os primeiros `window_days` do acervo com os últimos.

        Usar o início do acervo como referência é uma aproximação assumida: o
        ideal é gravar a distribuição de treino junto do modelo no registry e
        comparar contra ela. Enquanto o registry não guarda esse artefato, esta
        proxy detecta a mesma classe de problema — e o teste que importa
        (feature saiu do suporte) não depende da escolha.
        """
        start, end = self._readings.available_range()
        span = timedelta(days=self._window_days)

        if end - start < 2 * span:
            raise InsufficientDataError(
                required=2 * self._window_days,
                available=(end - start).days,
                subject="dias de histórico",
            )

        reference = self._readings.readings_between(start, start + span)
        current = self._readings.readings_between(end - span, end)

        if not reference or not current:
            raise InsufficientDataError(
                required=1,
                available=min(len(reference), len(current)),
                subject="leituras nos períodos comparados",
            )

        return DriftReport.of(
            {
                feature: population_stability_index(
                    reference=[reading.feature(feature) for reading in reference],
                    current=[reading.feature(feature) for reading in current],
                    bins=self._bins,
                )
                for feature in self._features
            }
        )
