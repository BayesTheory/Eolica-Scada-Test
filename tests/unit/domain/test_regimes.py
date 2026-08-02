"""Regime operacional.

Este value object existe por causa de uma medição. O detector que comparava cada
janela contra a distribuição de toda a operação normal entregou, sobre 61.239
janelas do dataset completo, 42% de precisão e 34% de taxa de alarme falso.

A explicação: uma turbina saudável a 2 m/s e a 11 m/s são estados diferentes.
Comparar os dois contra a mesma referência faz o erro de reconstrução medir
vento, não saúde — e como o vento varia o dia inteiro, o erro fica alto em
blocos sustentados, que é exatamente o padrão observado.

Condicionar ao regime levou a precisão a 98,8% e a taxa de alarme falso a 0,20%.
"""

from __future__ import annotations

from eolica.domain.turbine import OperatingRegime, TurbineSpec, WindSpeed

SPEC = TurbineSpec.aventa_av7()  # cut-in 2.0, cut-out 12.0


class TestClassificacao:
    def test_vento_calmo_fica_abaixo_do_cut_in(self) -> None:
        assert OperatingRegime.of(WindSpeed(0.5), SPEC) is OperatingRegime.BELOW_CUT_IN

    def test_exatamente_no_cut_in_ja_e_carga_parcial(self) -> None:
        assert OperatingRegime.of(WindSpeed(2.0), SPEC) is OperatingRegime.PARTIAL_LOAD

    def test_logo_abaixo_do_cut_in_ainda_nao_produz(self) -> None:
        assert OperatingRegime.of(WindSpeed(1.99), SPEC) is OperatingRegime.BELOW_CUT_IN

    def test_faixa_intermediaria_e_carga_parcial(self) -> None:
        assert OperatingRegime.of(WindSpeed(5.0), SPEC) is OperatingRegime.PARTIAL_LOAD

    def test_acima_da_nominal_e_carga_plena(self) -> None:
        """Nominal ≈ 2.0 + 0.55 × 10.0 = 7.5 m/s."""
        assert OperatingRegime.of(WindSpeed(9.0), SPEC) is OperatingRegime.FULL_LOAD

    def test_exatamente_no_cut_out_ainda_opera(self) -> None:
        assert OperatingRegime.of(WindSpeed(12.0), SPEC) is OperatingRegime.FULL_LOAD

    def test_acima_do_cut_out_a_turbina_para(self) -> None:
        """Parar acima do cut-out é o comportamento correto, não anomalia."""
        assert OperatingRegime.of(WindSpeed(12.1), SPEC) is OperatingRegime.ABOVE_CUT_OUT

    def test_vento_extremo_do_dataset(self) -> None:
        """O dataset real chega a 18,5 m/s."""
        assert OperatingRegime.of(WindSpeed(18.5), SPEC) is OperatingRegime.ABOVE_CUT_OUT


class TestProdutividade:
    def test_apenas_as_faixas_de_carga_produzem(self) -> None:
        assert OperatingRegime.PARTIAL_LOAD.is_productive is True
        assert OperatingRegime.FULL_LOAD.is_productive is True

    def test_fora_da_faixa_nao_se_espera_geracao(self) -> None:
        assert OperatingRegime.BELOW_CUT_IN.is_productive is False
        assert OperatingRegime.ABOVE_CUT_OUT.is_productive is False


class TestFronteiras:
    def test_fronteiras_vem_da_folha_de_dados_e_nao_dos_quantis(self) -> None:
        """Uma spec diferente move as fronteiras de forma previsível.

        Derivar as faixas de quantis dos dados faria o regime significar coisas
        diferentes antes e depois de um retreino — e um período atipicamente
        calmo deslocaria as fronteiras sem que a máquina tivesse mudado.
        """
        larger = TurbineSpec(rated_power_kw=2000.0, cut_in_mps=3.0, cut_out_mps=25.0)
        assert OperatingRegime.of(WindSpeed(2.5), larger) is OperatingRegime.BELOW_CUT_IN
        assert OperatingRegime.of(WindSpeed(2.5), SPEC) is OperatingRegime.PARTIAL_LOAD

    def test_cobre_toda_a_reta_sem_buraco(self) -> None:
        for tenth in range(0, 300):
            wind = WindSpeed(tenth / 10)
            assert isinstance(OperatingRegime.of(wind, SPEC), OperatingRegime)
