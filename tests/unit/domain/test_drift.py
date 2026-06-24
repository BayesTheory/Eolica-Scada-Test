"""Subdomínio `monitoring`: detecção de drift entre a distribuição de treino e
a que está chegando em produção.

Este contexto não existia no v1 — e é o que responde "o modelo ainda vale?".
Um autoencoder treinado nos dados de 2022 continua devolvendo erros de
reconstrução em 2024 sem reclamar de nada; a única forma de saber que o mundo
mudou embaixo dele é comparar as distribuições.

PSI e KS são implementados em Python puro (o domínio não importa scipy) e
testados contra valores conhecidos.
"""

from __future__ import annotations

import pytest

from eolica.domain.monitoring import (
    DriftMethod,
    DriftReport,
    DriftSeverity,
    kolmogorov_smirnov,
    population_stability_index,
)
from eolica.shared.errors import InsufficientDataError, InvalidValueError


class TestPopulationStabilityIndex:
    def test_distribuicoes_identicas_dao_psi_zero(self) -> None:
        values = [float(i) for i in range(100)]
        score = population_stability_index(reference=values, current=values, bins=10)
        assert score.value == pytest.approx(0.0, abs=1e-9)
        assert score.method is DriftMethod.PSI

    def test_deslocamento_total_da_psi_severo(self) -> None:
        reference = [float(i) for i in range(100)]
        current = [float(i) for i in range(1000, 1100)]
        score = population_stability_index(reference=reference, current=current, bins=10)
        assert score.value > 0.25
        assert score.severity is DriftSeverity.SEVERE

    def test_deslocamento_leve_fica_abaixo_do_limiar_de_acao(self) -> None:
        reference = [float(i) for i in range(1000)]
        current = [float(i) + 1.0 for i in range(1000)]
        score = population_stability_index(reference=reference, current=current, bins=10)
        assert score.severity is DriftSeverity.NONE

    def test_classificacao_de_severidade_segue_a_convencao_do_setor(self) -> None:
        """< 0.10 estável | 0.10–0.25 investigar | > 0.25 agir."""
        assert DriftSeverity.classify(0.05) is DriftSeverity.NONE
        assert DriftSeverity.classify(0.10) is DriftSeverity.MODERATE
        assert DriftSeverity.classify(0.20) is DriftSeverity.MODERATE
        assert DriftSeverity.classify(0.25) is DriftSeverity.SEVERE
        assert DriftSeverity.classify(0.90) is DriftSeverity.SEVERE

    def test_referencia_vazia_falha(self) -> None:
        with pytest.raises(InsufficientDataError):
            population_stability_index(reference=[], current=[1.0], bins=10)

    def test_amostra_atual_vazia_falha(self) -> None:
        with pytest.raises(InsufficientDataError):
            population_stability_index(reference=[1.0], current=[], bins=10)

    def test_rejeita_numero_de_bins_invalido(self) -> None:
        with pytest.raises(InvalidValueError, match="bins"):
            population_stability_index(reference=[1.0, 2.0], current=[1.0], bins=1)

    def test_referencia_constante_nao_divide_por_zero(self) -> None:
        """Uma feature travada (sensor morto) tem variância zero.

        É um caso real: `PitchDeg` fica em 14.034 por longos períodos. Sem
        tratamento, os limites de bin colapsam e o cálculo estoura.
        """
        score = population_stability_index(reference=[5.0] * 50, current=[5.0] * 50, bins=10)
        assert score.value == pytest.approx(0.0, abs=1e-9)

    def test_feature_travada_que_passa_a_variar_acusa_drift(self) -> None:
        score = population_stability_index(
            reference=[5.0] * 50, current=[float(i) for i in range(50)], bins=10
        )
        assert score.severity is DriftSeverity.SEVERE


class TestKolmogorovSmirnov:
    def test_amostras_identicas_dao_zero(self) -> None:
        values = [float(i) for i in range(50)]
        score = kolmogorov_smirnov(reference=values, current=values)
        assert score.value == pytest.approx(0.0, abs=1e-12)
        assert score.method is DriftMethod.KS

    def test_amostras_disjuntas_dao_um(self) -> None:
        score = kolmogorov_smirnov(
            reference=[float(i) for i in range(100)],
            current=[float(i) for i in range(100, 200)],
        )
        assert score.value == pytest.approx(1.0)

    def test_deslocamento_de_metade_da_distribuicao(self) -> None:
        """Metade da massa deslocada para fora do suporte de referência."""
        score = kolmogorov_smirnov(
            reference=[float(i) for i in range(100)],
            current=[float(i) for i in range(50)] + [float(i) for i in range(200, 250)],
        )
        assert score.value == pytest.approx(0.5, abs=0.02)

    def test_amostra_vazia_falha(self) -> None:
        with pytest.raises(InsufficientDataError):
            kolmogorov_smirnov(reference=[], current=[1.0])


class TestDriftReport:
    def test_agrega_por_feature_e_reporta_a_pior(self) -> None:
        report = DriftReport.of(
            {
                "wind_speed": population_stability_index(
                    reference=[float(i) for i in range(100)],
                    current=[float(i) for i in range(100)],
                    bins=10,
                ),
                "generator_temperature": population_stability_index(
                    reference=[float(i) for i in range(100)],
                    current=[float(i) for i in range(1000, 1100)],
                    bins=10,
                ),
            }
        )
        assert report.severity is DriftSeverity.SEVERE
        assert report.worst_feature == "generator_temperature"

    def test_relatorio_sem_drift_e_estavel(self) -> None:
        values = [float(i) for i in range(100)]
        report = DriftReport.of(
            {"wind_speed": population_stability_index(reference=values, current=values, bins=10)}
        )
        assert report.severity is DriftSeverity.NONE
        assert report.requires_action is False

    def test_relatorio_severo_exige_acao(self) -> None:
        report = DriftReport.of(
            {
                "power": population_stability_index(
                    reference=[float(i) for i in range(100)],
                    current=[float(i) for i in range(1000, 1100)],
                    bins=10,
                )
            }
        )
        assert report.requires_action is True

    def test_rejeita_relatorio_vazio(self) -> None:
        with pytest.raises(InsufficientDataError):
            DriftReport.of({})
