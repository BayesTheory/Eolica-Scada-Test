"""Subdomínio `forecasting`: projeção de geração de potência."""

from eolica.domain.forecasting.ports import PowerForecastModel
from eolica.domain.forecasting.value_objects import Horizon, PowerForecast

__all__ = ["Horizon", "PowerForecast", "PowerForecastModel"]
