"""Compara detectores de anomalia pelo mesmo backtest, sobre o mesmo dado.

Existe para responder empiricamente à pergunta que o primeiro backtest deixou
aberta: o detector z-score global tem 34% de taxa de alarme falso porque o
método é ruim, ou porque ele compara cada janela contra a distribuição de toda a
operação normal — misturando vento de 2 m/s com vento de 11 m/s?

Uso:
    python scripts/compare_detectors.py
"""

from __future__ import annotations

import sys
import time
from datetime import timedelta

from eolica.application.use_cases import BacktestDetector
from eolica.domain.health import AnomalyThreshold, ReconstructionError
from eolica.domain.turbine import TurbineSpec
from eolica.infrastructure.config import PROJECT_ROOT, Settings
from eolica.infrastructure.ml.baselines import ZScoreBaselineDetector, calibrate_threshold_windows
from eolica.infrastructure.ml.regime_detector import RegimeConditionedDetector
from eolica.infrastructure.persistence import CsvScadaRepository
from eolica.shared.errors import EolicaError

WINDOW_SIZE = 60
PERSISTENCE_WINDOWS = (1, 6)


def evaluate(name: str, detector: object, repository: CsvScadaRepository, windows: list) -> None:
    started = time.perf_counter()

    errors: list[ReconstructionError] = []
    for window in windows:
        errors.extend(detector.reconstruction_errors(window))  # type: ignore[attr-defined]

    threshold = AnomalyThreshold.from_percentile(errors, percentile=99.5)

    summary = BacktestDetector(
        readings=repository,
        health_model=detector,  # type: ignore[arg-type]
        threshold=threshold,
        sampling_interval=timedelta(minutes=10),
        persistence_windows=PERSISTENCE_WINDOWS,
    ).execute()

    elapsed = time.perf_counter() - started
    print(f"\n{'=' * 74}")
    print(f"{name}")
    print(f"{'=' * 74}")
    print(f"  limiar calibrado ........ {threshold.value:.4f}")
    print(f"  janelas de referência ... {len(errors):,}")
    print(f"  tempo ................... {elapsed:.1f}s")
    print()
    header = (
        f"  {'janela':>7} {'episódios':>10} {'precisão':>9} "
        f"{'recall':>8} {'alarme falso':>13} {'F1':>7}"
    )
    print(header)
    print(f"  {'-' * 60}")
    for outcome in summary.report.outcomes:
        print(
            f"  {outcome.persistence_window:>7} {outcome.episodes:>10,} "
            f"{outcome.metrics.precision:>8.1%} {outcome.recall:>7.1%} "
            f"{outcome.false_alarm_rate:>12.2%} {outcome.metrics.f1:>6.1%}"
        )


def main() -> int:
    settings = Settings(
        data_path=PROJECT_ROOT / "data" / "processed" / "scada_resampled_10min_base.csv"
    )
    if not settings.data_path.exists():
        print(f"ERRO: dataset completo ausente em {settings.data_path}", file=sys.stderr)
        return 1

    repository = CsvScadaRepository.from_path(settings.data_path)
    reference = repository.normal_operation_readings()
    windows = calibrate_threshold_windows(
        reference, window_size=WINDOW_SIZE, sampling_interval=timedelta(minutes=10)
    )

    print(f"dataset ....... {len(repository):,} leituras")
    print(f"referência .... {len(reference):,} em operação normal, {len(windows)} segmentos")
    print(f"janela ........ {WINDOW_SIZE} passos ({WINDOW_SIZE * 10} minutos)")

    spec = TurbineSpec.aventa_av7()

    evaluate(
        "A — z-score GLOBAL (uma distribuição para toda a operação normal)",
        ZScoreBaselineDetector.fit(windows, window_size=WINDOW_SIZE),
        repository,
        windows,
    )

    conditioned = RegimeConditionedDetector.fit(windows, window_size=WINDOW_SIZE, spec=spec)
    evaluate(
        f"B — z-score CONDICIONADO ao regime ({len(conditioned.calibrated_regimes)} regimes: "
        f"{', '.join(r.value for r in conditioned.calibrated_regimes)})",
        conditioned,
        repository,
        windows,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EolicaError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
