"""Value objects e entidades do subdomínio `turbine`.

Estes testes fixam três invariantes que o serviço v1 deixava soltas:

1. Potência negativa é um estado físico legítimo (consumo parasita) e precisa
   ser distinguida do valor exibido ao operador. No v1 essa regra vivia dentro
   do prompt do LLM ("REGRA DA POTÊNCIA NEGATIVA: (...) NUNCA mostre um valor
   de potência negativo"), ou seja: o modelo de linguagem era o guardião de uma
   invariante de negócio.
2. Código de status desconhecido não pode ser silenciosamente tratado como
   "não é 10, logo é falha". O dataset tem o código 305, indocumentado.
3. Uma janela de leituras não pode atravessar um buraco temporal. O dataset
   real tem 30 gaps, dois deles de mais de 24h; o v1 fatiava com `iloc[-n:]` e
   entregava ao modelo uma janela que cruzava um dia inteiro de ausência.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from eolica.domain.turbine import (
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
from eolica.shared.errors import InsufficientDataError, InvalidValueError

# ─────────────────────────────────────────────────────────────────────────────
# WindSpeed
# ─────────────────────────────────────────────────────────────────────────────


class TestWindSpeed:
    def test_aceita_valor_valido(self) -> None:
        assert WindSpeed(7.5).mps == 7.5

    def test_zero_e_valido(self) -> None:
        assert WindSpeed(0.0).mps == 0.0

    def test_rejeita_negativo(self) -> None:
        with pytest.raises(InvalidValueError, match="não pode ser negativa"):
            WindSpeed(-0.1)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejeita_nao_finito(self, bad: float) -> None:
        with pytest.raises(InvalidValueError, match="finita"):
            WindSpeed(bad)

    def test_e_comparavel_e_hashavel(self) -> None:
        assert WindSpeed(3.0) < WindSpeed(4.0)
        assert len({WindSpeed(3.0), WindSpeed(3.0)}) == 1


# ─────────────────────────────────────────────────────────────────────────────
# PowerKw — onde a regra que estava no prompt do LLM passa a viver
# ─────────────────────────────────────────────────────────────────────────────


class TestPowerKw:
    def test_preserva_valor_negativo_medido(self) -> None:
        """O dado bruto não é adulterado: -0.09 kW é consumo real da turbina."""
        assert PowerKw(-0.09).kw == pytest.approx(-0.09)

    def test_valor_exibido_faz_clamp_do_negativo(self) -> None:
        assert PowerKw(-0.09).for_display() == 0.0

    def test_valor_exibido_preserva_o_positivo(self) -> None:
        assert PowerKw(3.4).for_display() == pytest.approx(3.4)

    def test_identifica_consumo_parasita(self) -> None:
        assert PowerKw(-0.05).is_parasitic is True
        assert PowerKw(0.0).is_parasitic is False

    def test_rejeita_nao_finito(self) -> None:
        with pytest.raises(InvalidValueError, match="finita"):
            PowerKw(float("nan"))


# ─────────────────────────────────────────────────────────────────────────────
# OperatingStatus
# ─────────────────────────────────────────────────────────────────────────────


class TestOperatingStatus:
    def test_reconhece_producao(self) -> None:
        assert OperatingStatus.from_code(10) is OperatingStatus.PRODUCING

    def test_reconhece_falha(self) -> None:
        assert OperatingStatus.from_code(13) is OperatingStatus.FAULT

    def test_aceita_codigo_como_float(self) -> None:
        """O CSV traz o status como float (10.0) por causa do resample."""
        assert OperatingStatus.from_code(10.0) is OperatingStatus.PRODUCING

    @pytest.mark.parametrize("code", [8, 9, 11, 12, 305])
    def test_codigo_indocumentado_vira_unknown_e_nao_falha(self, code: int) -> None:
        """Estes códigos existem no dataset mas não constam do metadado do
        fabricante. Chutar semântica seria pior que admitir desconhecimento."""
        assert OperatingStatus.from_code(code) is OperatingStatus.UNKNOWN

    def test_apenas_producao_conta_como_operacao_saudavel(self) -> None:
        assert OperatingStatus.PRODUCING.is_healthy_operation is True
        assert OperatingStatus.FAULT.is_healthy_operation is False
        assert OperatingStatus.UNKNOWN.is_healthy_operation is False


# ─────────────────────────────────────────────────────────────────────────────
# TurbineSpec
# ─────────────────────────────────────────────────────────────────────────────


class TestTurbineSpec:
    def test_spec_do_aventa_av7(self) -> None:
        spec = TurbineSpec.aventa_av7()
        assert spec.rated_power_kw == pytest.approx(6.2)
        assert spec.cut_in_mps == pytest.approx(2.0)
        assert spec.cut_out_mps == pytest.approx(12.0)

    def test_vento_abaixo_do_cut_in_nao_produz(self) -> None:
        spec = TurbineSpec.aventa_av7()
        assert spec.expects_production(WindSpeed(1.9)) is False

    def test_vento_na_faixa_produz(self) -> None:
        spec = TurbineSpec.aventa_av7()
        assert spec.expects_production(WindSpeed(6.0)) is True

    def test_vento_acima_do_cut_out_nao_produz(self) -> None:
        """Acima do cut-out a turbina protege a si mesma e para."""
        spec = TurbineSpec.aventa_av7()
        assert spec.expects_production(WindSpeed(12.1)) is False

    def test_rejeita_cut_in_maior_que_cut_out(self) -> None:
        with pytest.raises(InvalidValueError, match="cut-in"):
            TurbineSpec(rated_power_kw=6.2, cut_in_mps=13.0, cut_out_mps=12.0)


# ─────────────────────────────────────────────────────────────────────────────
# ReadingWindow — a correção do bug de janela cega
# ─────────────────────────────────────────────────────────────────────────────

BASE_TIME = datetime(2022, 1, 14, 0, 0, tzinfo=UTC)
STEP = timedelta(minutes=10)


def _reading(offset_steps: int, *, power: float = 1.0) -> TurbineReading:
    return TurbineReading(
        timestamp=BASE_TIME + offset_steps * STEP,
        wind_speed=WindSpeed(5.0),
        power=PowerKw(power),
        rotor_speed=RotorSpeed(30.0),
        generator_temperature=Temperature(40.0),
        pitch=PitchAngle(20.0),
        status=OperatingStatus.PRODUCING,
    )


class TestReadingWindow:
    def test_aceita_sequencia_contigua(self) -> None:
        window = ReadingWindow.of([_reading(i) for i in range(6)], expected_interval=STEP)
        assert len(window) == 6

    def test_rejeita_janela_vazia(self) -> None:
        with pytest.raises(InsufficientDataError):
            ReadingWindow.of([], expected_interval=STEP)

    def test_rejeita_timestamps_fora_de_ordem(self) -> None:
        readings = [_reading(0), _reading(2), _reading(1)]
        with pytest.raises(InvalidValueError, match="ordem cronológica"):
            ReadingWindow.of(readings, expected_interval=STEP)

    def test_rejeita_timestamps_duplicados(self) -> None:
        with pytest.raises(InvalidValueError, match="ordem cronológica"):
            ReadingWindow.of([_reading(0), _reading(0)], expected_interval=STEP)

    def test_rejeita_janela_que_atravessa_gap(self) -> None:
        """Este é o bug do v1: `iloc[-n:]` não sabe que faltam horas no meio."""
        readings = [_reading(0), _reading(1), _reading(50)]
        with pytest.raises(InvalidValueError, match="descontinuidade"):
            ReadingWindow.of(readings, expected_interval=STEP)

    def test_segmenta_em_janelas_contiguas_ao_inves_de_falhar(self) -> None:
        """A recuperação correta não é abortar: é fatiar nos buracos."""
        readings = [_reading(0), _reading(1), _reading(2), _reading(50), _reading(51)]
        segments = ReadingWindow.split_on_gaps(readings, expected_interval=STEP)
        assert [len(s) for s in segments] == [3, 2]

    def test_segmentacao_descarta_segmentos_menores_que_o_minimo(self) -> None:
        readings = [_reading(0), _reading(1), _reading(2), _reading(50)]
        segments = ReadingWindow.split_on_gaps(readings, expected_interval=STEP, min_length=3)
        assert [len(s) for s in segments] == [3]

    def test_expoe_o_intervalo_coberto(self) -> None:
        window = ReadingWindow.of([_reading(i) for i in range(4)], expected_interval=STEP)
        assert window.start == BASE_TIME
        assert window.end == BASE_TIME + 3 * STEP

    def test_extrai_serie_de_uma_feature(self) -> None:
        window = ReadingWindow.of(
            [_reading(i, power=float(i)) for i in range(3)], expected_interval=STEP
        )
        assert window.series("power") == (0.0, 1.0, 2.0)

    def test_serie_de_feature_inexistente_falha_alto(self) -> None:
        window = ReadingWindow.of([_reading(0)], expected_interval=STEP)
        with pytest.raises(InvalidValueError, match="desconhecida"):
            window.series("nao_existe")

    def test_e_imutavel(self) -> None:
        window = ReadingWindow.of([_reading(0)], expected_interval=STEP)
        with pytest.raises((AttributeError, TypeError)):
            window.readings = ()  # type: ignore[misc]
