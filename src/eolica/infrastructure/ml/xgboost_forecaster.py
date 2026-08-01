"""Adaptador XGBoost para previsão de potência. Requer o extra `[ml]`.

A diferença estrutural em relação à v1 é que este adaptador **não sabe montar
features**. Ele recebe uma `LagFeatureView` e delega — a mesma view usada no
treino. É o que torna o skew impossível em vez de improvável.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import xgboost as xgb

from eolica.domain.forecasting import Horizon, PowerForecast
from eolica.domain.turbine import PowerKw
from eolica.infrastructure.featurestore import FeatureView
from eolica.shared.errors import ConfigurationError

if TYPE_CHECKING:
    from eolica.domain.turbine import ReadingWindow


class XGBoostPowerForecaster:
    """Satisfaz `domain.forecasting.ports.PowerForecastModel`."""

    def __init__(
        self,
        *,
        booster: xgb.XGBRegressor,
        # `FeatureView` e não `LagFeatureView`: o previsor aceita qualquer
        # composição de features — lag, janela móvel ou as duas juntas — desde
        # que a assinatura bata com a do treino.
        view: FeatureView,
        version: str = "xgboost@local",
        trained_signature: str | None = None,
    ) -> None:
        """
        Args:
            trained_signature: assinatura da view usada no treino. Se informada
                e diferente da view atual, a construção falha. É a proteção
                contra o bug silencioso da v1: mudar `n_lags` sem retreinar
                passava despercebido e só degradava a previsão.
        """
        if trained_signature is not None and trained_signature != view.signature:
            raise ConfigurationError(
                "A feature view não corresponde à usada no treino do modelo. "
                "Retreine o modelo ou restaure a configuração de features.",
                trained=trained_signature,
                current=view.signature,
            )
        self._booster = booster
        self._view = view
        self._version = version

    @property
    def required_history(self) -> int:
        """Delegado à view — o modelo não tem opinião própria sobre isso."""
        return self._view.required_history

    @property
    def version(self) -> str:
        return self._version

    @property
    def feature_signature(self) -> str:
        return self._view.signature

    def predict(self, window: ReadingWindow, horizon: Horizon) -> PowerForecast:
        """Prevê a potência no passo seguinte ao fim da janela."""
        history = _window_to_frame(window, self._view.source_columns)
        features = self._view.build_inference_vector(history)
        prediction = float(self._booster.predict(features)[0])

        return PowerForecast(
            power=PowerKw(prediction),
            issued_at=window.end,
            horizon=horizon,
            model_version=self._version,
        )


def _window_to_frame(window: ReadingWindow, features: tuple[str, ...]) -> pd.DataFrame:
    """Converte uma janela de domínio no DataFrame que a view espera.

    A conversão acontece aqui, na infraestrutura, e não no domínio — que não
    conhece pandas.
    """
    names = tuple(sorted(features))
    return pd.DataFrame(
        {name: window.series(name) for name in names},
        index=pd.DatetimeIndex([reading.timestamp for reading in window.readings]),
    )
