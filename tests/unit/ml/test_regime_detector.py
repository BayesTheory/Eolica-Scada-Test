"""Detector condicionado ao regime.

O teste que carrega o argumento é `test_variacao_de_vento_nao_gera_anomalia`:
ele reproduz, em miniatura, a causa dos 34% de alarme falso do detector global.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from eolica.domain.health import ReconstructionModel
from eolica.domain.turbine import (
    OperatingRegime,
    OperatingStatus,
    PitchAngle,
    PowerKw,
    ReadingWindow,
    RotorSpeed,
    Temperature,
    TurbineReading,
    TurbineSpec,
    WindSpeed,
)
from eolica.infrastructure.ml.baselines import ZScoreBaselineDetector
from eolica.infrastructure.ml.regime_detector import (
    MIN_SAMPLES_PER_REGIME,
    RegimeConditionedDetector,
)
from eolica.shared.errors import InsufficientDataError

STEP = timedelta(minutes=10)
SPEC = TurbineSpec.aventa_av7()
BASE = datetime(2022, 1, 14, tzinfo=UTC)


RATED_WIND = 7.5  # 2.0 + 0.55 × (12.0 − 2.0)


def _reading(index: int, *, wind: float, temperature: float = 40.0) -> TurbineReading:
    """Leitura fisicamente coerente com a curva de potência da máquina.

    Potência cresce com o cubo do vento até a nominal de 6,2 kW; o pitch fica
    mínimo em carga parcial e abre em carga plena para descartar potência
    excedente. Sem isso, os regimes não teriam assinaturas distintas e o
    fixture não representaria uma turbina.
    """
    productive = SPEC.cut_in_mps <= wind <= SPEC.cut_out_mps
    # Jitter determinístico: sem variância intra-regime, o desvio padrão do
    # regime seria zero e a normalização vira subtração pura.
    jitter = ((index * 37) % 11 - 5) / 100.0

    if not productive:
        return TurbineReading(
            timestamp=BASE + index * STEP,
            wind_speed=WindSpeed(max(0.0, wind + jitter)),
            power=PowerKw(0.0 + jitter / 10),
            rotor_speed=RotorSpeed(0.0),
            generator_temperature=Temperature(temperature + jitter),
            pitch=PitchAngle(80.0 + jitter),
            status=OperatingStatus.PRODUCING,
        )

    fraction = min(1.0, (wind / RATED_WIND) ** 3)
    return TurbineReading(
        timestamp=BASE + index * STEP,
        wind_speed=WindSpeed(wind + jitter),
        power=PowerKw(6.2 * fraction + jitter / 10),
        rotor_speed=RotorSpeed(min(65.0, 6.0 * wind) + jitter),
        generator_temperature=Temperature(temperature + jitter),
        pitch=PitchAngle((14.0 if wind < RATED_WIND else 25.0) + jitter),
        status=OperatingStatus.PRODUCING,
    )


def _window(winds: list[float], *, temperature: float = 40.0) -> ReadingWindow:
    return ReadingWindow.of(
        [_reading(i, wind=w, temperature=temperature) for i, w in enumerate(winds)],
        expected_interval=STEP,
    )


def _reference_windows() -> list[ReadingWindow]:
    """Operação normal com a **assimetria** que o dado real tem.

    A primeira versão deste fixture dava peso igual aos três regimes — e com
    isso nenhum ficava longe da média global, então o detector global não errava
    e o teste não reproduzia o efeito observado em produção.

    O dataset real tem mediana de vento de 1,91 m/s e máximo de 18,5: a maior
    parte do tempo é calmaria, e vento forte é um estado raro e distante da
    média. É essa assimetria que faz um modelo único tratar operação em carga
    plena como desvio — e é ela que o fixture precisa ter para testar a coisa
    certa.
    """
    return [
        _window([1.0] * 2800),  # ~69% — abaixo do cut-in
        _window([5.0] * 1000),  # ~25% — carga parcial
        _window([10.0] * 250),  # ~6%  — carga plena
    ]


@pytest.fixture(scope="module")
def reference_windows() -> list[ReadingWindow]:
    return _reference_windows()


@pytest.fixture(scope="module")
def detector(reference_windows: list[ReadingWindow]) -> RegimeConditionedDetector:
    return RegimeConditionedDetector.fit(reference_windows, window_size=6, spec=SPEC)


class TestCalibracao:
    def test_satisfaz_a_porta_do_dominio(self, detector: RegimeConditionedDetector) -> None:
        assert isinstance(detector, ReconstructionModel)

    def test_calibra_uma_referencia_por_regime(self, detector: RegimeConditionedDetector) -> None:
        assert set(detector.calibrated_regimes) == {
            OperatingRegime.BELOW_CUT_IN,
            OperatingRegime.PARTIAL_LOAD,
            OperatingRegime.FULL_LOAD,
        }

    def test_regime_com_poucas_amostras_cai_no_fallback(self) -> None:
        """Calibrar um regime com vinte pontos é pior que usar a referência
        genérica — a estatística seria ruído."""
        windows = [
            _window([5.0] * (MIN_SAMPLES_PER_REGIME + 10)),
            _window([10.0] * 20),  # abaixo do mínimo
        ]
        detector = RegimeConditionedDetector.fit(windows, window_size=6, spec=SPEC)
        assert OperatingRegime.FULL_LOAD not in detector.calibrated_regimes
        assert OperatingRegime.PARTIAL_LOAD in detector.calibrated_regimes

    def test_referencia_vazia_falha(self) -> None:
        with pytest.raises(InsufficientDataError):
            RegimeConditionedDetector.fit([], window_size=6, spec=SPEC)

    def test_versao_declara_quantos_regimes_foram_calibrados(
        self, detector: RegimeConditionedDetector
    ) -> None:
        assert "3regimes" in detector.version


class TestPontuacao:
    def test_produz_um_erro_por_sub_janela(self, detector: RegimeConditionedDetector) -> None:
        assert len(detector.reconstruction_errors(_window([5.0] * 20))) == 15

    def test_janela_curta_demais_nao_produz_erro(self, detector: RegimeConditionedDetector) -> None:
        assert detector.reconstruction_errors(_window([5.0] * 3)) == []

    def test_erros_sao_nao_negativos(self, detector: RegimeConditionedDetector) -> None:
        errors = detector.reconstruction_errors(_window([5.0] * 20))
        assert all(e.value >= 0 for e in errors)


class TestOArgumentoCentral:
    """A causa dos 34% de alarme falso, reproduzida em miniatura."""

    def test_variacao_de_vento_nao_gera_anomalia(self, detector: RegimeConditionedDetector) -> None:
        """Uma turbina saudável operando em vento alto não é uma anomalia.

        O detector global compara vento de 10 m/s contra uma distribuição que
        mistura calmaria e vendaval — e acusa. O condicionado compara contra o
        que se espera *naquele regime*, e não acusa.
        """
        calm = detector.reconstruction_errors(_window([1.0] * 20))
        windy = detector.reconstruction_errors(_window([10.0] * 20))

        worst_calm = max(e.value for e in calm)
        worst_windy = max(e.value for e in windy)
        assert worst_windy < worst_calm * 5, (
            "operar em regime diferente não pode produzir erro de outra ordem de grandeza"
        )

    def test_detector_global_confunde_regime_com_anomalia(
        self, reference_windows: list[ReadingWindow]
    ) -> None:
        """O contraste que justifica a mudança.

        Mesmo dado, mesmos dois regimes: o detector global pontua o vento alto
        com erro muito maior que o vento calmo, embora ambos sejam operação
        perfeitamente saudável. É o mecanismo por trás dos 34% de taxa de alarme
        falso medidos no dataset completo.
        """
        global_detector = ZScoreBaselineDetector.fit(reference_windows, window_size=6)
        calm = max(e.value for e in global_detector.reconstruction_errors(_window([1.0] * 20)))
        windy = max(e.value for e in global_detector.reconstruction_errors(_window([10.0] * 20)))
        assert windy > calm * 5, "é o comportamento que a medição no dataset real expôs"

    def test_anomalia_real_dentro_do_regime_continua_visivel(
        self, detector: RegimeConditionedDetector
    ) -> None:
        """A contrapartida: condicionar não pode cegar o detector.

        Mesma condição de vento, temperatura de gerador muito acima do normal —
        isso tem que pontuar alto.
        """
        healthy = detector.reconstruction_errors(_window([5.0] * 20, temperature=40.0))
        overheating = detector.reconstruction_errors(_window([5.0] * 20, temperature=95.0))
        assert max(e.value for e in overheating) > max(e.value for e in healthy) * 10
