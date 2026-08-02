"""Composition root: onde as dependências concretas são amarradas.

Este é o único módulo do projeto que conhece *ao mesmo tempo* o domínio, os
casos de uso e os adaptadores. Todo o resto depende só de abstrações — é o que
permite trocar o CSV por um banco, ou o baseline pelo LSTM, editando apenas
aqui.

Contraste com o v1, onde a composição acontecia no nível de módulo do
`inference_api.py`: carregar o CSV, conectar no MLflow e instanciar modelos
eram efeitos colaterais do `import`. Isso significava que `import inference_api`
num teste tentava abrir socket para o MLflow e chamava `sys.exit(1)` se falhasse
— derrubando o processo do pytest antes de qualquer asserção.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta

from eolica.application.use_cases import BacktestDetector, CheckDrift, GenerateDailyReport
from eolica.domain.forecasting import Horizon, PowerForecastModel
from eolica.domain.health import AnomalyThreshold, ReconstructionError, ReconstructionModel
from eolica.infrastructure.config import Settings
from eolica.infrastructure.ml.baselines import (
    MovingAverageForecaster,
    ZScoreBaselineDetector,
    calibrate_threshold_windows,
)
from eolica.infrastructure.persistence import CsvScadaRepository
from eolica.shared.errors import DataSourceError, InsufficientDataError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Calibration:
    """O que a calibração produziu — exposto em `/health/ready` e nos logs."""

    threshold: AnomalyThreshold
    reference_windows: int
    reference_errors: int
    duration_seconds: float


@dataclass(slots=True)
class Container:
    """As dependências vivas do processo."""

    settings: Settings
    repository: CsvScadaRepository
    health_model: ReconstructionModel
    forecast_model: PowerForecastModel
    calibration: Calibration
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def sampling_interval(self) -> timedelta:
        return timedelta(minutes=self.settings.sampling_interval_minutes)

    @property
    def is_ready(self) -> bool:
        return all(self.checks.values())

    def daily_report_use_case(self) -> GenerateDailyReport:
        return GenerateDailyReport(
            readings=self.repository,
            health_model=self.health_model,
            forecast_model=self.forecast_model,
            threshold=self.calibration.threshold,
            persistence_window=self.settings.persistence_window,
            sampling_interval=self.sampling_interval,
            forecast_horizon=Horizon(
                steps=self.settings.forecast_horizon_steps, step=self.sampling_interval
            ),
        )

    def drift_use_case(self) -> CheckDrift:
        return CheckDrift(
            readings=self.repository,
            bins=self.settings.drift_bins,
            window_days=self.settings.drift_reference_days,
        )

    def backtest_use_case(self) -> BacktestDetector:
        return BacktestDetector(
            readings=self.repository,
            health_model=self.health_model,
            threshold=self.calibration.threshold,
            sampling_interval=self.sampling_interval,
        )


def build_container(settings: Settings) -> Container:
    """Monta o container, calibrando o detector na subida do processo.

    A calibração do limiar acontece **aqui**, uma vez, e não dentro do handler
    HTTP como no v1 — onde a primeira requisição do dia varria 32 mil registros
    enquanto o cliente esperava, e duas requisições concorrentes escreviam no
    mesmo atributo mutável.
    """
    started = time.perf_counter()
    repository = _load_repository(settings)

    reference_readings = repository.normal_operation_readings()
    windows = calibrate_threshold_windows(
        reference_readings,
        window_size=settings.health_window_size,
        sampling_interval=timedelta(minutes=settings.sampling_interval_minutes),
    )
    if not windows:
        raise InsufficientDataError(
            required=settings.health_window_size,
            available=len(reference_readings),
            subject="leituras contíguas em operação normal",
        )

    detector = ZScoreBaselineDetector.fit(windows, window_size=settings.health_window_size)

    errors: list[ReconstructionError] = []
    for window in windows:
        errors.extend(detector.reconstruction_errors(window))

    threshold = AnomalyThreshold.from_percentile(
        errors, percentile=settings.anomaly_threshold_percentile
    )
    elapsed = time.perf_counter() - started

    logger.info(
        "calibração concluída",
        extra={
            "threshold": threshold.value,
            "reference_windows": len(windows),
            "reference_errors": len(errors),
            "duration_seconds": round(elapsed, 3),
        },
    )

    container = Container(
        settings=settings,
        repository=repository,
        health_model=detector,
        forecast_model=MovingAverageForecaster(required_history=settings.forecast_n_lags),
        calibration=Calibration(
            threshold=threshold,
            reference_windows=len(windows),
            reference_errors=len(errors),
            duration_seconds=elapsed,
        ),
    )
    container.checks = {
        "repository_loaded": len(repository) > 0,
        "health_model_calibrated": len(errors) > 0,
        "forecast_model_loaded": True,
    }
    return container


def _load_repository(settings: Settings) -> CsvScadaRepository:
    """Carrega a telemetria, com fallback explícito para o sample versionado.

    `data/processed/` é gerado pelo pipeline de ingestão e não vai para o git.
    Num clone limpo o arquivo não existe — e o comportamento certo é subir com o
    sample e **dizer isso alto no log**, não estourar. Fallback silencioso seria
    pior que falhar: alguém acabaria olhando métricas de duas semanas de dado
    achando que eram de dois anos.
    """
    if settings.data_path.exists():
        return CsvScadaRepository.from_path(settings.data_path)

    if not settings.sample_data_path.exists():
        raise DataSourceError(
            "Nem o dataset processado nem o sample foram encontrados",
            data_path=str(settings.data_path),
            sample_path=str(settings.sample_data_path),
        )

    logger.warning(
        "dataset processado ausente; subindo com o SAMPLE versionado (2 semanas de dados). "
        "Rode `eolica ingest` para gerar o dataset completo.",
        extra={"expected": str(settings.data_path), "using": str(settings.sample_data_path)},
    )
    return CsvScadaRepository.from_path(settings.sample_data_path)
