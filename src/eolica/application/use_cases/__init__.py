"""Casos de uso do sistema."""

from eolica.application.use_cases.backtest_detector import BacktestDetector, BacktestSummary
from eolica.application.use_cases.check_drift import CheckDrift
from eolica.application.use_cases.generate_daily_report import (
    DailyReport,
    DataCoverage,
    GenerateDailyReport,
)

__all__ = [
    "BacktestDetector",
    "BacktestSummary",
    "CheckDrift",
    "DailyReport",
    "DataCoverage",
    "GenerateDailyReport",
]
