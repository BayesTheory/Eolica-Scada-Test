"""Subdomínio `health`: threshold de anomalia e veredito de saúde.

Duas regras de negócio são repatriadas aqui:

**1. O critério de "em manutenção".** No v1 vivia na regra 4 do prompt do LLM:

    "Se 'anomalias_detectadas' for maior que 0 E 'anomalias_dia_anterior'
     também for maior que 0, o status da turbina é 'EM MANUTENÇÃO'.
     Ignore o status 'ALERTA' vindo da API nesse caso."

Ou seja: a API dizia ALERTA, e pedia-se ao modelo de linguagem que a
contradissesse. A regra valia só no chat, era invisível para qualquer outro
consumidor da API, e ninguém conseguia testá-la.

**2. A janela de persistência.** `config.yaml` declarava `persistence_window: 6`
e **nenhuma linha do código a lia**. O detector marcava anomalia em qualquer
janela isolada acima do limiar — num sinal de 10 em 10 minutos, isso é ruído de
sensor virando alarme. Seis janelas consecutivas = uma hora de desvio sustentado.
"""

from __future__ import annotations

import pytest

from eolica.domain.health import (
    AnomalyThreshold,
    HealthStatus,
    ReconstructionError,
    ThresholdMethod,
    evaluate_health,
)
from eolica.shared.errors import InsufficientDataError, InvalidValueError


def _errors(*values: float) -> list[ReconstructionError]:
    return [ReconstructionError(v) for v in values]


# ─────────────────────────────────────────────────────────────────────────────
# ReconstructionError
# ─────────────────────────────────────────────────────────────────────────────


class TestReconstructionError:
    def test_aceita_erro_positivo(self) -> None:
        assert ReconstructionError(0.42).value == pytest.approx(0.42)

    def test_aceita_zero(self) -> None:
        assert ReconstructionError(0.0).value == 0.0

    def test_rejeita_negativo(self) -> None:
        """MSE é uma média de quadrados: negativo é bug de cálculo."""
        with pytest.raises(InvalidValueError, match="não pode ser negativo"):
            ReconstructionError(-0.1)

    def test_rejeita_nan(self) -> None:
        with pytest.raises(InvalidValueError, match="finito"):
            ReconstructionError(float("nan"))


# ─────────────────────────────────────────────────────────────────────────────
# AnomalyThreshold
# ─────────────────────────────────────────────────────────────────────────────


class TestAnomalyThreshold:
    def test_percentil_reproduz_a_interpolacao_do_numpy(self) -> None:
        """Compatível com `np.percentile(..., method='linear')`.

        Importa porque o limiar do v1 era calculado com numpy no serving; se a
        reimplementação em Python puro divergisse, o limiar mudaria de valor
        numa refatoração supostamente sem efeito comportamental.
        """
        threshold = AnomalyThreshold.from_percentile(_errors(1, 2, 3, 4, 5), percentile=99.5)
        assert threshold.value == pytest.approx(4.98)

    def test_percentil_50_e_a_mediana(self) -> None:
        threshold = AnomalyThreshold.from_percentile(_errors(1, 2, 3, 4, 5), percentile=50)
        assert threshold.value == pytest.approx(3.0)

    def test_percentil_registra_o_metodo_usado(self) -> None:
        threshold = AnomalyThreshold.from_percentile(_errors(1, 2, 3), percentile=99.5)
        assert threshold.method is ThresholdMethod.PERCENTILE
        assert threshold.parameter == pytest.approx(99.5)

    def test_desvios_padrao_usa_populacional_como_o_numpy(self) -> None:
        """mean + n·σ, com σ populacional (ddof=0), igual a `np.std`."""
        threshold = AnomalyThreshold.from_std_deviations(_errors(1, 2, 3, 4, 5), n_std=3.0)
        assert threshold.value == pytest.approx(3.0 + 3.0 * 1.4142135, rel=1e-6)
        assert threshold.method is ThresholdMethod.STD_DEVIATIONS

    def test_rejeita_amostra_vazia(self) -> None:
        with pytest.raises(InsufficientDataError):
            AnomalyThreshold.from_percentile([], percentile=99.5)

    def test_rejeita_percentil_fora_da_faixa(self) -> None:
        with pytest.raises(InvalidValueError, match="percentil"):
            AnomalyThreshold.from_percentile(_errors(1, 2), percentile=101)

    def test_classifica_erro_acima_do_limiar(self) -> None:
        threshold = AnomalyThreshold.from_percentile(_errors(1, 2, 3, 4, 5), percentile=50)
        assert threshold.is_exceeded_by(ReconstructionError(3.5)) is True
        assert threshold.is_exceeded_by(ReconstructionError(2.5)) is False

    def test_limiar_e_imutavel(self) -> None:
        threshold = AnomalyThreshold.from_percentile(_errors(1, 2), percentile=50)
        with pytest.raises((AttributeError, TypeError)):
            threshold.value = 99.0  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_health — persistência
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD = AnomalyThreshold(value=1.0, method=ThresholdMethod.PERCENTILE, parameter=99.5)


class TestPersistencia:
    def test_tudo_abaixo_do_limiar_e_ok(self) -> None:
        verdict = evaluate_health(
            errors=_errors(0.1, 0.2, 0.1, 0.3), threshold=THRESHOLD, persistence_window=3
        )
        assert verdict.status is HealthStatus.OK
        assert verdict.sustained_anomalies == 0

    def test_pico_isolado_nao_gera_alerta(self) -> None:
        """Um outlier de 10 minutos é ruído de sensor, não falha de turbina."""
        verdict = evaluate_health(
            errors=_errors(0.1, 5.0, 0.1, 0.1), threshold=THRESHOLD, persistence_window=3
        )
        assert verdict.status is HealthStatus.OK
        assert verdict.exceedances == 1, "o pico é contado..."
        assert verdict.sustained_anomalies == 0, "...mas não sustentado"

    def test_sequencia_menor_que_a_janela_nao_gera_alerta(self) -> None:
        verdict = evaluate_health(
            errors=_errors(0.1, 5.0, 5.0, 0.1), threshold=THRESHOLD, persistence_window=3
        )
        assert verdict.status is HealthStatus.OK

    def test_sequencia_do_tamanho_da_janela_gera_alerta(self) -> None:
        verdict = evaluate_health(
            errors=_errors(0.1, 5.0, 5.0, 5.0, 0.1), threshold=THRESHOLD, persistence_window=3
        )
        assert verdict.status is HealthStatus.ALERT
        assert verdict.sustained_anomalies == 3

    def test_conta_apenas_a_corrida_sustentada(self) -> None:
        """Dois picos isolados + uma corrida de 3: só a corrida conta."""
        verdict = evaluate_health(
            errors=_errors(5.0, 0.1, 5.0, 0.1, 5.0, 5.0, 5.0),
            threshold=THRESHOLD,
            persistence_window=3,
        )
        assert verdict.exceedances == 5
        assert verdict.sustained_anomalies == 3

    def test_janela_de_persistencia_1_alerta_em_qualquer_pico(self) -> None:
        verdict = evaluate_health(
            errors=_errors(0.1, 5.0, 0.1), threshold=THRESHOLD, persistence_window=1
        )
        assert verdict.status is HealthStatus.ALERT

    def test_rejeita_janela_de_persistencia_invalida(self) -> None:
        with pytest.raises(InvalidValueError, match="persistência"):
            evaluate_health(errors=_errors(0.1), threshold=THRESHOLD, persistence_window=0)

    def test_rejeita_lista_de_erros_vazia(self) -> None:
        with pytest.raises(InsufficientDataError):
            evaluate_health(errors=[], threshold=THRESHOLD, persistence_window=3)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_health — a regra de manutenção que morava no prompt do LLM
# ─────────────────────────────────────────────────────────────────────────────


class TestRegraDeManutencao:
    def test_anomalia_sustentada_hoje_e_ontem_e_manutencao(self) -> None:
        verdict = evaluate_health(
            errors=_errors(5.0, 5.0, 5.0),
            threshold=THRESHOLD,
            persistence_window=3,
            previous_period_anomalies=7,
        )
        assert verdict.status is HealthStatus.UNDER_MAINTENANCE

    def test_anomalia_sustentada_so_hoje_e_alerta(self) -> None:
        verdict = evaluate_health(
            errors=_errors(5.0, 5.0, 5.0),
            threshold=THRESHOLD,
            persistence_window=3,
            previous_period_anomalies=0,
        )
        assert verdict.status is HealthStatus.ALERT

    def test_sem_dado_do_dia_anterior_nao_conclui_manutencao(self) -> None:
        """Ausência de informação não é evidência de manutenção.

        O v1 mandava -1 quando a análise do dia anterior falhava, e o prompt
        comparava `> 0` — então uma falha de processamento virava, corretamente
        por acidente, "não é manutenção". Aqui a ignorância é explícita.
        """
        verdict = evaluate_health(
            errors=_errors(5.0, 5.0, 5.0),
            threshold=THRESHOLD,
            persistence_window=3,
            previous_period_anomalies=None,
        )
        assert verdict.status is HealthStatus.ALERT
        assert verdict.previous_period_known is False

    def test_dia_anterior_ruim_mas_hoje_ok_nao_e_manutencao(self) -> None:
        verdict = evaluate_health(
            errors=_errors(0.1, 0.1, 0.1),
            threshold=THRESHOLD,
            persistence_window=3,
            previous_period_anomalies=99,
        )
        assert verdict.status is HealthStatus.OK

    def test_rejeita_contagem_anterior_negativa(self) -> None:
        """-1 como sentinela de erro é exatamente o que se quer proibir."""
        with pytest.raises(InvalidValueError, match="negativa"):
            evaluate_health(
                errors=_errors(0.1),
                threshold=THRESHOLD,
                persistence_window=1,
                previous_period_anomalies=-1,
            )


class TestVerdict:
    def test_veredito_carrega_o_limiar_usado(self) -> None:
        """Auditabilidade: reproduzir a decisão exige saber o limiar."""
        verdict = evaluate_health(
            errors=_errors(0.1, 0.2), threshold=THRESHOLD, persistence_window=2
        )
        assert verdict.threshold == THRESHOLD
        assert verdict.evaluated_windows == 2

    def test_veredito_explica_a_decisao(self) -> None:
        verdict = evaluate_health(
            errors=_errors(5.0, 5.0), threshold=THRESHOLD, persistence_window=2
        )
        assert "sustentada" in verdict.reason.lower()

    def test_veredito_e_imutavel(self) -> None:
        verdict = evaluate_health(errors=_errors(0.1), threshold=THRESHOLD, persistence_window=1)
        with pytest.raises((AttributeError, TypeError)):
            verdict.status = HealthStatus.ALERT  # type: ignore[misc]
