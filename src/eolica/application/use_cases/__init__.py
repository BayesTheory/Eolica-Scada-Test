"""Casos de uso do sistema."""

from eolica.application.use_cases.backtest_detector import BacktestDetector, BacktestSummary
from eolica.application.use_cases.check_drift import CheckDrift
from eolica.application.use_cases.generate_daily_report import (
    DailyReport,
    DataCoverage,
    GenerateDailyReport,
)
from eolica.application.use_cases.summarise_coverage import (
    CoverageSummary,
    DayCoverage,
    SummariseCoverage,
)

__all__ = [
    "BacktestDetector",
    "BacktestSummary",
    "CheckDrift",
    "CoverageSummary",
    "DailyReport",
    "DataCoverage",
    "DayCoverage",
    "GenerateDailyReport",
    "SummariseCoverage",
]
