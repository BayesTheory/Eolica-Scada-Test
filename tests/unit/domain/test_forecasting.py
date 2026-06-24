"""Subdomínio `forecasting`: previsão de geração de potência."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from eolica.domain.forecasting import Horizon, PowerForecast
from eolica.domain.turbine import PowerKw, TurbineSpec
from eolica.shared.errors import InvalidValueError

ISSUED_AT = datetime(2022, 1, 14, 12, 0, tzinfo=UTC)
STEP = timedelta(minutes=10)


class TestHorizon:
    def test_um_passo_a_frente(self) -> None:
        horizon = Horizon(steps=1, step=STEP)
        assert horizon.duration == timedelta(minutes=10)

    def test_multiplos_passos(self) -> None:
        assert Horizon(steps=6, step=STEP).duration == timedelta(hours=1)

    def test_rejeita_zero_passos(self) -> None:
        with pytest.raises(InvalidValueError, match="pelo menos 1"):
            Horizon(steps=0, step=STEP)

    def test_rejeita_passo_nao_positivo(self) -> None:
        with pytest.raises(InvalidValueError, match="positiva"):
            Horizon(steps=1, step=timedelta(0))


class TestPowerForecast:
    def _forecast(self, kw: float) -> PowerForecast:
        return PowerForecast(
            power=PowerKw(kw),
            issued_at=ISSUED_AT,
            horizon=Horizon(steps=1, step=STEP),
            model_version="xgboost@3",
        )

    def test_calcula_o_instante_alvo(self) -> None:
        assert self._forecast(2.0).target_time == ISSUED_AT + STEP

    def test_previsao_negativa_e_preservada_no_dado(self) -> None:
        """O modelo pode prever negativo; o dado bruto não é adulterado."""
        assert self._forecast(-0.3).power.kw == pytest.approx(-0.3)

    def test_previsao_negativa_e_zerada_na_exibicao(self) -> None:
        """A mesma regra do PowerKw, agora aplicada à previsão.

        No v1 esta era a regra 3 do prompt do LLM. Um cliente da API que não
        fosse o chat recebia `-0.3 kW` e não tinha como saber que aquilo devia
        ser exibido como zero.
        """
        assert self._forecast(-0.3).for_display() == 0.0

    def test_registra_a_versao_do_modelo(self) -> None:
        """Sem isto não há como reproduzir uma previsão passada."""
        assert self._forecast(1.0).model_version == "xgboost@3"

    def test_detecta_previsao_acima_da_potencia_nominal(self) -> None:
        """6.2 kW é o teto físico da Aventa AV-7: prever 50 kW é bug, não vento."""
        spec = TurbineSpec.aventa_av7()
        assert self._forecast(50.0).exceeds_rated(spec) is True
        assert self._forecast(5.0).exceeds_rated(spec) is False

    def test_e_imutavel(self) -> None:
        forecast = self._forecast(1.0)
        with pytest.raises((AttributeError, TypeError)):
            forecast.power = PowerKw(9.0)  # type: ignore[misc]
