"""Casos de uso do sistema."""

from eolica.application.use_cases.generate_daily_report import (
    DailyReport,
    DataCoverage,
    GenerateDailyReport,
)

__all__ = ["DailyReport", "DataCoverage", "GenerateDailyReport"]
