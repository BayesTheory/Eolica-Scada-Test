"""Implementações em memória das portas, para testar casos de uso sem I/O.

São fakes, não mocks: têm comportamento real e verificável. Um mock que devolve
`Mock()` para tudo passa em qualquer asserção e não prova nada — foi assim que o
v1 chegou a produção com zero testes de integração e um `KeyError` não tratado
no caminho mais quente da API.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta

from eolica.domain.forecasting import Horizon, PowerForecast
from eolica.domain.health import ReconstructionError
from eolica.domain.turbine import (
    OperatingStatus,
    PitchAngle,
    PowerKw,
    ReadingWindow,
    RotorSpeed,
    Temperature,
    TurbineReading,
    WindSpeed,
)
from eolica.shared.errors import InsufficientDataError, NotFoundError

SAMPLING_INTERVAL = timedelta(minutes=10)


def make_reading(
    moment: datetime,
    *,
    power: float = 1.0,
    wind: float = 5.0,
    temperature: float = 40.0,
    status: OperatingStatus = OperatingStatus.PRODUCING,
) -> TurbineReading:
    return TurbineReading(
        timestamp=moment,
        wind_speed=WindSpeed(wind),
        power=PowerKw(power),
        rotor_speed=RotorSpeed(30.0),
        generator_temperature=Temperature(temperature),
        pitch=PitchAngle(20.0),
        status=status,
    )


def make_day(day: date, *, count: int = 144, start_step: int = 0) -> list[TurbineReading]:
    """Um dia de leituras contíguas na grade de 10 minutos."""
    midnight = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    return [
        make_reading(midnight + (start_step + i) * SAMPLING_INTERVAL, power=float(i % 7))
        for i in range(count)
    ]


class InMemoryScadaRepository:
    """`ScadaRepository` sobre uma lista ordenada de leituras."""

    def __init__(self, readings: Sequence[TurbineReading]) -> None:
        self._readings = sorted(readings, key=lambda r: r.timestamp)
        self._timestamps = [r.timestamp for r in self._readings]
        self._days = {r.timestamp.date() for r in self._readings}

    def readings_for_day(self, day: date) -> Sequence[TurbineReading]:
        if day not in self._days:
            raise NotFoundError("Telemetria", day.isoformat())
        return [r for r in self._readings if r.timestamp.date() == day]

    def readings_before(self, moment: datetime, *, limit: int) -> Sequence[TurbineReading]:
        cut = bisect.bisect_right(self._timestamps, moment)
        window = self._readings[max(0, cut - limit) : cut]
        if len(window) < limit:
            raise InsufficientDataError(
                required=limit, available=len(window), subject="observações"
            )
        return window

    def readings_between(self, start: datetime, end: datetime) -> Sequence[TurbineReading]:
        return [r for r in self._readings if start <= r.timestamp <= end]

    def available_range(self) -> tuple[datetime, datetime]:
        if not self._readings:
            raise NotFoundError("Acervo", "vazio")
        return self._timestamps[0], self._timestamps[-1]


class FakeReconstructionModel:
    """`ReconstructionModel` que devolve erros programados pelo teste."""

    def __init__(
        self,
        *,
        window_size: int = 6,
        error_for: Callable[[int], float] | None = None,
    ) -> None:
        self._window_size = window_size
        self._error_for = error_for or (lambda _: 0.1)
        self.calls: list[ReadingWindow] = []

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("generator_temperature", "pitch", "power", "rotor_speed", "wind_speed")

    def reconstruction_errors(self, window: ReadingWindow) -> Sequence[ReconstructionError]:
        self.calls.append(window)
        count = len(window) - self._window_size + 1
        if count < 1:
            return []
        return [ReconstructionError(self._error_for(i)) for i in range(count)]


class FakeForecastModel:
    """`PowerForecastModel` determinístico."""

    def __init__(self, *, kw: float = 2.5, required_history: int = 6) -> None:
        self._kw = kw
        self._required_history = required_history

    @property
    def required_history(self) -> int:
        return self._required_history

    @property
    def version(self) -> str:
        return "fake-forecaster@1"

    def predict(self, window: ReadingWindow, horizon: Horizon) -> PowerForecast:
        return PowerForecast(
            power=PowerKw(self._kw),
            issued_at=window.end,
            horizon=horizon,
            model_version=self.version,
        )


class BrokenForecastModel(FakeForecastModel):
    """Modelo de previsão que sempre falha — para exercitar a degradação."""

    def predict(self, window: ReadingWindow, horizon: Horizon) -> PowerForecast:
        raise InsufficientDataError(required=99, available=1, subject="observações")


class FrozenClock:
    """`Clock` parado num instante."""

    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


class RecordingMetrics:
    """`MetricsRecorder` que só guarda o que recebeu."""

    def __init__(self) -> None:
        self.inferences: list[tuple[str, float, str]] = []
        self.verdicts: list[str] = []
        self.drifts: list[tuple[str, float]] = []

    def record_inference(self, *, model: str, duration_seconds: float, outcome: str) -> None:
        self.inferences.append((model, duration_seconds, outcome))

    def record_health_verdict(self, *, status: str) -> None:
        self.verdicts.append(status)

    def record_drift(self, *, feature: str, score: float) -> None:
        self.drifts.append((feature, score))
