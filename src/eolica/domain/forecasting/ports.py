"""Portas do subdomínio `forecasting`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from eolica.domain.forecasting.value_objects import Horizon, PowerForecast
    from eolica.domain.turbine import ReadingWindow


@runtime_checkable
class PowerForecastModel(Protocol):
    """Um modelo que projeta potência a partir do histórico recente."""

    @property
    def required_history(self) -> int:
        """Leituras mínimas necessárias para montar as features de entrada.

        No v1 este número (`n_lags`) era lido de dois caminhos diferentes do
        `config.yaml` — `params.n_lags` no treino e `n_lags` no serving — e
        coincidia em 6 apenas porque nenhum dos dois existia no arquivo e ambos
        caíam no mesmo default. Aqui o modelo declara o que precisa.
        """
        ...

    @property
    def version(self) -> str:
        """Identidade do modelo carregado, para rastrear a previsão."""
        ...

    def predict(self, window: ReadingWindow, horizon: Horizon) -> PowerForecast: ...
