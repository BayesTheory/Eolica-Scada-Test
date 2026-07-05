"""Modelos baseline: a régua contra a qual o LSTM e o XGBoost são julgados.

Todo projeto de ML precisa de um baseline burro e explícito. Sem ele, "o
autoencoder atingiu MSE 0.003" não significa nada — pode ser excelente ou pode
ser pior que prever a média.

O v1 não tinha nenhum. Estes dois são deliberadamente triviais, implementados
sem torch nem xgboost, e servem a três propósitos:

1. régua de comparação nas métricas de treino;
2. permitir que a API suba e responda de verdade num ambiente sem GPU e sem
   MLflow — o que torna os testes de integração e a demo possíveis;
3. fallback operacional se o registry estiver fora do ar.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from datetime import timedelta

from eolica.domain.forecasting import Horizon, PowerForecast
from eolica.domain.health import ReconstructionError
from eolica.domain.turbine import FEATURE_NAMES, PowerKw, ReadingWindow
from eolica.shared.errors import ConfigurationError, InsufficientDataError


class ZScoreBaselineDetector:
    """Detector de anomalia por desvio padronizado — o baseline de saúde.

    Ajusta média e desvio de cada feature sobre janelas de operação normal e
    pontua cada sub-janela pelo z² médio. É o método mais simples que ainda
    responde à pergunta certa ("isto se parece com o que eu vi quando estava
    saudável?"), e um autoencoder que não o supere não justifica seu custo.

    Satisfaz `domain.health.ports.ReconstructionModel` sem herdar nada dele.
    """

    def __init__(
        self,
        *,
        window_size: int,
        feature_names: Sequence[str] = FEATURE_NAMES,
        means: Sequence[float],
        deviations: Sequence[float],
    ) -> None:
        if not (len(feature_names) == len(means) == len(deviations)):
            raise ConfigurationError(
                "Features, médias e desvios devem ter o mesmo comprimento",
                features=len(feature_names),
                means=len(means),
                deviations=len(deviations),
            )
        if window_size < 1:
            raise ConfigurationError("window_size deve ser positivo", window_size=window_size)

        self._window_size = window_size
        self._feature_names = tuple(feature_names)
        self._means = tuple(means)
        # Desvio zero (sensor travado) viraria divisão por zero; 1.0 neutraliza
        # a normalização daquela feature em vez de derrubar a inferência.
        self._deviations = tuple(d if d > 0 else 1.0 for d in deviations)

    @classmethod
    def fit(
        cls,
        windows: Sequence[ReadingWindow],
        *,
        window_size: int,
        feature_names: Sequence[str] = FEATURE_NAMES,
    ) -> ZScoreBaselineDetector:
        """Calibra sobre janelas de referência (operação normal)."""
        columns: list[list[float]] = [[] for _ in feature_names]
        for window in windows:
            for row in window.matrix(feature_names):
                for index, value in enumerate(row):
                    columns[index].append(value)

        if not columns[0]:
            raise InsufficientDataError(required=1, available=0, subject="leituras de referência")

        return cls(
            window_size=window_size,
            feature_names=feature_names,
            means=[statistics.fmean(column) for column in columns],
            deviations=[
                statistics.pstdev(column) if len(column) > 1 else 0.0 for column in columns
            ],
        )

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def version(self) -> str:
        return "zscore-baseline@1"

    def reconstruction_errors(self, window: ReadingWindow) -> list[ReconstructionError]:
        """z² médio de cada sub-janela deslizante.

        Calculado com soma de prefixos: O(n) em vez do O(n·w) da soma ingênua
        por janela. Com 32 mil leituras de referência e janela de 60 passos, a
        diferença é entre ~0,1 s e mais de 10 s na calibração — que roda na
        subida do processo e portanto entra direto no tempo de deploy.
        """
        rows = window.matrix(self._feature_names)
        count = len(rows) - self._window_size + 1
        if count < 1:
            return []

        row_totals = [
            sum(
                ((value - self._means[i]) / self._deviations[i]) ** 2 for i, value in enumerate(row)
            )
            for row in rows
        ]

        prefix = [0.0]
        for total in row_totals:
            prefix.append(prefix[-1] + total)

        cell_count = self._window_size * len(self._feature_names)
        return [
            ReconstructionError((prefix[start + self._window_size] - prefix[start]) / cell_count)
            for start in range(count)
        ]


class PersistenceForecaster:
    """Previsão por persistência: ŷ(t+1) = y(t). O baseline de previsão.

    Em séries de vento a persistência é notoriamente difícil de bater em
    horizontes curtos. Um XGBoost com R² alto que não supere isto num teste
    honesto está aprendendo a autocorrelação e nada mais — e é exatamente o que
    o `avg_r2_score` do v1, medido sem comparação, não conseguia revelar.
    """

    def __init__(self, *, required_history: int = 1) -> None:
        if required_history < 1:
            raise ConfigurationError("required_history deve ser positivo")
        self._required_history = required_history

    @property
    def required_history(self) -> int:
        return self._required_history

    @property
    def version(self) -> str:
        return "persistence-baseline@1"

    def predict(self, window: ReadingWindow, horizon: Horizon) -> PowerForecast:
        if len(window) < self._required_history:
            raise InsufficientDataError(
                required=self._required_history, available=len(window), subject="observações"
            )
        return PowerForecast(
            power=PowerKw(window.readings[-1].power.kw),
            issued_at=window.end,
            horizon=horizon,
            model_version=self.version,
        )


class MovingAverageForecaster(PersistenceForecaster):
    """Média móvel das últimas N leituras — baseline levemente menos ingênuo."""

    @property
    def version(self) -> str:
        return f"moving-average-{self._required_history}@1"

    def predict(self, window: ReadingWindow, horizon: Horizon) -> PowerForecast:
        if len(window) < self._required_history:
            raise InsufficientDataError(
                required=self._required_history, available=len(window), subject="observações"
            )
        recent = window.series("power")[-self._required_history :]
        return PowerForecast(
            power=PowerKw(statistics.fmean(recent)),
            issued_at=window.end,
            horizon=horizon,
            model_version=self.version,
        )


def calibrate_threshold_windows(
    readings: Sequence[object],
    *,
    window_size: int,
    sampling_interval: timedelta,
) -> list[ReadingWindow]:
    """Fatia leituras de referência em janelas contíguas utilizáveis.

    Usado tanto para ajustar o baseline quanto para calcular o limiar de
    anomalia — as duas coisas precisam ver exatamente o mesmo recorte, senão o
    limiar é calibrado sobre uma distribuição que o detector nunca viu.
    """
    return ReadingWindow.split_on_gaps(
        readings,  # type: ignore[arg-type]
        expected_interval=sampling_interval,
        min_length=window_size,
    )


def is_close(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    """Comparação de ponto flutuante para asserções de teste e verificação."""
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)
