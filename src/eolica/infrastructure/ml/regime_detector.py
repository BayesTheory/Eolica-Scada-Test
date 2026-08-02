"""Detector condicionado ao regime operacional.

Nasce de uma medição, não de uma intuição. O backtest do detector z-score global
sobre 61.239 janelas devolveu precisão de 42% e taxa de alarme falso de 34% — e
mostrou que ajustar a janela de persistência não muda nada, porque o erro não é
composto de picos isolados, é ruído sustentado.

Ruído sustentado tem uma explicação natural: o detector estava comparando cada
janela contra a distribuição de **toda** a operação normal. Mas operar a 2 m/s e
operar a 11 m/s são estados legitimamente diferentes da mesma máquina saudável.
Um modelo único faz o erro de reconstrução medir *vento*, não *saúde* — e como o
vento varia o dia inteiro, o erro fica alto o tempo todo, em blocos.

Aqui cada regime tem sua própria referência, e a pontuação de uma janela é
sempre contra o regime em que ela ocorreu. Um limiar global continua funcionando
porque z-scores já são adimensionais e comparáveis entre regimes.

Se isto ajuda é pergunta empírica, respondida rodando o mesmo backtest.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from eolica.domain.health import ReconstructionError
from eolica.domain.turbine import (
    FEATURE_NAMES,
    OperatingRegime,
    ReadingWindow,
    TurbineSpec,
    WindSpeed,
)
from eolica.shared.errors import ConfigurationError, InsufficientDataError

# Amostras mínimas para uma referência de regime ser confiável. Abaixo disso o
# regime cai no fallback global — melhor uma referência genérica que uma
# calibrada em vinte pontos.
MIN_SAMPLES_PER_REGIME = 200


@dataclass(frozen=True, slots=True)
class RegimeReference:
    """Média e desvio de cada feature, dentro de um regime."""

    means: tuple[float, ...]
    deviations: tuple[float, ...]
    samples: int

    @classmethod
    def of(cls, columns: Sequence[Sequence[float]]) -> RegimeReference:
        return cls(
            means=tuple(statistics.fmean(column) for column in columns),
            # Desvio zero (feature travada dentro do regime) neutraliza a
            # normalização daquela feature em vez de estourar a divisão.
            deviations=tuple(
                statistics.pstdev(column)
                if len(column) > 1 and statistics.pstdev(column) > 0
                else 1.0
                for column in columns
            ),
            samples=len(columns[0]) if columns else 0,
        )


class RegimeConditionedDetector:
    """Satisfaz `domain.health.ports.ReconstructionModel`, por regime."""

    def __init__(
        self,
        *,
        window_size: int,
        spec: TurbineSpec,
        references: dict[OperatingRegime, RegimeReference],
        fallback: RegimeReference,
        feature_names: Sequence[str] = FEATURE_NAMES,
    ) -> None:
        if window_size < 1:
            raise ConfigurationError("window_size deve ser positivo", window_size=window_size)
        self._window_size = window_size
        self._spec = spec
        self._references = references
        self._fallback = fallback
        self._feature_names = tuple(feature_names)
        self._wind_index = self._feature_names.index("wind_speed")

    @classmethod
    def fit(
        cls,
        windows: Sequence[ReadingWindow],
        *,
        window_size: int,
        spec: TurbineSpec,
        feature_names: Sequence[str] = FEATURE_NAMES,
    ) -> RegimeConditionedDetector:
        """Calibra uma referência por regime sobre janelas de operação normal."""
        names = tuple(feature_names)
        per_regime: dict[OperatingRegime, list[list[float]]] = {}
        everything: list[list[float]] = [[] for _ in names]

        for window in windows:
            for reading in window.readings:
                regime = OperatingRegime.of(reading.wind_speed, spec)
                bucket = per_regime.setdefault(regime, [[] for _ in names])
                for index, name in enumerate(names):
                    value = reading.feature(name)
                    bucket[index].append(value)
                    everything[index].append(value)

        if not everything[0]:
            raise InsufficientDataError(required=1, available=0, subject="leituras de referência")

        references = {
            regime: RegimeReference.of(columns)
            for regime, columns in per_regime.items()
            if len(columns[0]) >= MIN_SAMPLES_PER_REGIME
        }
        return cls(
            window_size=window_size,
            spec=spec,
            references=references,
            fallback=RegimeReference.of(everything),
            feature_names=names,
        )

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def version(self) -> str:
        return f"regime-zscore-baseline@1[{len(self._references)}regimes]"

    @property
    def calibrated_regimes(self) -> tuple[OperatingRegime, ...]:
        return tuple(sorted(self._references, key=lambda regime: regime.value))

    def reconstruction_errors(self, window: ReadingWindow) -> list[ReconstructionError]:
        """z² médio de cada sub-janela, contra a referência do seu regime.

        O regime de uma sub-janela é o do vento **mediano** dela, não o da
        média: numa janela que atravessa uma rajada, a mediana descreve melhor a
        condição predominante, e é o regime predominante que define o que é
        comportamento esperado ali.
        """
        rows = window.matrix(self._feature_names)
        count = len(rows) - self._window_size + 1
        if count < 1:
            return []

        errors: list[ReconstructionError] = []
        for start in range(count):
            chunk = rows[start : start + self._window_size]
            reference = self._reference_for(chunk)
            total = 0.0
            for row in chunk:
                for index, value in enumerate(row):
                    total += ((value - reference.means[index]) / reference.deviations[index]) ** 2
            errors.append(
                ReconstructionError(total / (self._window_size * len(self._feature_names)))
            )
        return errors

    def _reference_for(self, chunk: Sequence[Sequence[float]]) -> RegimeReference:
        median_wind = statistics.median(row[self._wind_index] for row in chunk)
        regime = OperatingRegime.of(WindSpeed(median_wind), self._spec)
        return self._references.get(regime, self._fallback)
