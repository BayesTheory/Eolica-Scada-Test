"""Caso de uso: relatório diário de saúde e previsão de uma turbina.

É o mesmo produto que o endpoint `/gerar_relatorio_diario/` do v1 entregava —
reescrito para que a lógica não dependa de FastAPI, de pandas nem do MLflow, e
para que cada caminho de erro tenha um significado.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo
from typing import TYPE_CHECKING

from eolica.domain.forecasting import Horizon, PowerForecast
from eolica.domain.health import (
    AnomalyThreshold,
    HealthVerdict,
    ReconstructionError,
    evaluate_health,
)
from eolica.domain.turbine import ReadingWindow
from eolica.shared.errors import EolicaError, InsufficientDataError, NotFoundError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eolica.application.ports import ScadaRepository
    from eolica.domain.forecasting import PowerForecastModel
    from eolica.domain.health import ReconstructionModel


@dataclass(frozen=True, slots=True)
class DataCoverage:
    """Quanto do dia foi efetivamente analisado.

    Campo novo em relação ao v1, e não é enfeite: um dia com 40% de cobertura
    produz um veredito "OK" que significa muito menos que um dia com 100%. Sem
    esse número, o operador não tem como calibrar a confiança no relatório.
    """

    readings: int
    expected_readings: int
    analysed_segments: int
    discarded_readings: int

    @property
    def completeness(self) -> float:
        """Fração do dia com medição, em [0, 1]."""
        if self.expected_readings == 0:
            return 0.0
        return min(1.0, self.readings / self.expected_readings)

    @property
    def is_fragmented(self) -> bool:
        """True se o dia teve descontinuidades."""
        return self.analysed_segments > 1


@dataclass(frozen=True, slots=True)
class DailyReport:
    """O produto do caso de uso."""

    day: date
    health: HealthVerdict
    coverage: DataCoverage
    forecast: PowerForecast | None
    forecast_unavailable_reason: str | None
    data_range: tuple[datetime, datetime]


class GenerateDailyReport:
    """Orquestra repositório, modelos e regras de domínio.

    Todas as colaborações chegam pelo construtor. Em particular o `threshold`:
    no v1 ele era calculado **dentro do handler HTTP**, na primeira requisição,
    varrendo os 32 mil registros de operação normal — e ficava guardado num
    atributo mutável compartilhado entre requisições. A primeira chamada do dia
    levava dezenas de segundos, e duas chamadas concorrentes disputavam o mesmo
    atributo. Aqui o limiar é um dado de entrada, calculado uma vez no
    `lifespan` da aplicação.
    """

    def __init__(
        self,
        *,
        readings: ScadaRepository,
        health_model: ReconstructionModel,
        forecast_model: PowerForecastModel,
        threshold: AnomalyThreshold,
        persistence_window: int,
        sampling_interval: timedelta,
        forecast_horizon: Horizon,
    ) -> None:
        self._readings = readings
        self._health_model = health_model
        self._forecast_model = forecast_model
        self._threshold = threshold
        self._persistence_window = persistence_window
        self._sampling_interval = sampling_interval
        self._forecast_horizon = forecast_horizon

    def execute(self, day: date) -> DailyReport:
        """Gera o relatório do dia.

        Raises:
            NotFoundError: o dia não está no acervo.
            InsufficientDataError: o dia existe mas nenhum trecho contíguo é
                longo o bastante para a janela do modelo.
        """
        readings = self._readings.readings_for_day(day)
        if not readings:
            raise NotFoundError("Telemetria", day.isoformat())

        errors, segments, analysed = self._reconstruction_errors(readings)
        if not errors:
            raise InsufficientDataError(
                required=self._health_model.window_size,
                available=max((len(s) for s in segments), default=0),
                subject="leituras contíguas",
            )

        verdict = evaluate_health(
            errors=errors,
            threshold=self._threshold,
            persistence_window=self._persistence_window,
            previous_period_anomalies=self._previous_day_anomalies(day),
        )

        forecast, reason = self._forecast_from(day)

        return DailyReport(
            day=day,
            health=verdict,
            coverage=DataCoverage(
                readings=len(readings),
                expected_readings=self._expected_readings_per_day(),
                analysed_segments=len(segments),
                discarded_readings=len(readings) - analysed,
            ),
            forecast=forecast,
            forecast_unavailable_reason=reason,
            data_range=self._readings.available_range(),
        )

    # ── passos ───────────────────────────────────────────────────────────────

    def _reconstruction_errors(
        self, readings: Sequence[object]
    ) -> tuple[list[ReconstructionError], list[ReadingWindow], int]:
        """Erros de reconstrução, respeitando as descontinuidades do dia.

        Um dia com um buraco de duas horas vira dois segmentos analisados
        separadamente — em vez de uma janela única que atravessa o buraco e
        produz erro de reconstrução artificialmente alto.
        """
        segments = ReadingWindow.split_on_gaps(
            readings,  # type: ignore[arg-type]
            expected_interval=self._sampling_interval,
            min_length=self._health_model.window_size,
        )

        errors: list[ReconstructionError] = []
        analysed = 0
        for segment in segments:
            errors.extend(self._health_model.reconstruction_errors(segment))
            analysed += len(segment)
        return errors, segments, analysed

    def _previous_day_anomalies(self, day: date) -> int | None:
        """Anomalias sustentadas na véspera, ou `None` se não dá para saber.

        `None` — e nunca o sentinela `-1` do v1 — é o que permite ao domínio
        distinguir "ontem foi limpo" de "não consegui olhar ontem".
        """
        try:
            previous = self._readings.readings_for_day(day - timedelta(days=1))
        except (NotFoundError, EolicaError):
            return None
        if not previous:
            return None

        errors, _, _ = self._reconstruction_errors(previous)
        if not errors:
            return None

        return evaluate_health(
            errors=errors,
            threshold=self._threshold,
            persistence_window=self._persistence_window,
        ).sustained_anomalies

    def _forecast_from(self, day: date) -> tuple[PowerForecast | None, str | None]:
        """Previsão para o passo seguinte ao fim do dia.

        Falha de previsão degrada o relatório, não o derruba: o veredito de
        saúde continua válido e é a parte crítica para o operador. Mas o motivo
        é reportado — o v1 devolvia a string `"Indisponível"` no lugar de um
        número, o que quebrava o tipo do campo e deixava o cliente sem saber
        por quê.
        """
        end_of_day = datetime.combine(day, time.max, tzinfo=self._sampling_tzinfo())
        needed = self._forecast_model.required_history
        try:
            history = self._readings.readings_before(end_of_day, limit=needed)
            window = ReadingWindow.of(history, expected_interval=self._sampling_interval)
            return self._forecast_model.predict(window, self._forecast_horizon), None
        except EolicaError as exc:
            return None, str(exc)

    def _sampling_tzinfo(self) -> tzinfo | None:
        """Fuso do acervo, para que o fim do dia seja comparável às leituras.

        Misturar datetime aware e naive levanta `TypeError` na comparação — um
        erro que só aparece em runtime e só no caminho da previsão.
        """
        start, _ = self._readings.available_range()
        return start.tzinfo

    def _expected_readings_per_day(self) -> int:
        return int(timedelta(days=1) / self._sampling_interval)
