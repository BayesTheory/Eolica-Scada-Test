"""Métricas e a regra que decide se um modelo merece ir para produção.

O v1 logava `avg_r2_score` no MLflow e registrava o modelo incondicionalmente —
`mlflow.register_model()` rodava logo depois do treino, sem comparação com nada.
Um R² de 0.87 não diz se o XGBoost aprendeu física de turbina ou apenas a
autocorrelação da série, que a previsão por persistência captura de graça.

Aqui a promoção é uma decisão de negócio com regra explícita: o desafiante
precisa superar o baseline por uma margem mínima, porque um modelo mais complexo
que empata custa mais para operar sem entregar mais.
"""

from __future__ import annotations

import pytest

from eolica.domain.evaluation import (
    DetectionMetrics,
    RegressionMetrics,
    compare_against_baseline,
)
from eolica.shared.errors import InsufficientDataError, InvalidValueError


class TestRegressionMetrics:
    def test_previsao_perfeita(self) -> None:
        metrics = RegressionMetrics.of(actual=[1.0, 2.0, 3.0], predicted=[1.0, 2.0, 3.0])
        assert metrics.rmse == pytest.approx(0.0)
        assert metrics.mae == pytest.approx(0.0)
        assert metrics.r2 == pytest.approx(1.0)

    def test_calcula_rmse_e_mae(self) -> None:
        metrics = RegressionMetrics.of(actual=[1.0, 2.0, 3.0], predicted=[2.0, 2.0, 2.0])
        assert metrics.rmse == pytest.approx((2 / 3) ** 0.5)
        assert metrics.mae == pytest.approx(2 / 3)

    def test_r2_de_um_modelo_que_so_preve_a_media_e_zero(self) -> None:
        actual = [1.0, 2.0, 3.0, 4.0]
        metrics = RegressionMetrics.of(actual=actual, predicted=[2.5] * 4)
        assert metrics.r2 == pytest.approx(0.0)

    def test_r2_negativo_para_modelo_pior_que_a_media(self) -> None:
        """R² negativo é informação valiosa e some quando ninguém compara."""
        metrics = RegressionMetrics.of(actual=[1.0, 2.0, 3.0], predicted=[10.0, 10.0, 10.0])
        assert metrics.r2 < 0

    def test_serie_constante_nao_divide_por_zero(self) -> None:
        """Variância zero no alvo torna o R² indefinido; reportamos NaN-safe."""
        metrics = RegressionMetrics.of(actual=[5.0, 5.0, 5.0], predicted=[5.0, 5.0, 5.0])
        assert metrics.r2 == pytest.approx(1.0)

    def test_rejeita_tamanhos_diferentes(self) -> None:
        with pytest.raises(InvalidValueError, match="mesmo tamanho"):
            RegressionMetrics.of(actual=[1.0, 2.0], predicted=[1.0])

    def test_rejeita_amostra_vazia(self) -> None:
        with pytest.raises(InsufficientDataError):
            RegressionMetrics.of(actual=[], predicted=[])


class TestDetectionMetrics:
    def test_deteccao_perfeita(self) -> None:
        metrics = DetectionMetrics.of(
            predicted=[True, True, False, False], actual=[True, True, False, False]
        )
        assert metrics.precision == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(1.0)
        assert metrics.f1 == pytest.approx(1.0)

    def test_taxa_de_alarme_falso(self) -> None:
        """3 negativos reais, 2 apontados como positivo -> 2/3."""
        metrics = DetectionMetrics.of(
            predicted=[True, True, False, True], actual=[False, False, False, True]
        )
        assert metrics.false_alarm_rate == pytest.approx(2 / 3)

    def test_nenhum_alarme_da_precisao_zero_sem_dividir_por_zero(self) -> None:
        metrics = DetectionMetrics.of(predicted=[False, False], actual=[True, False])
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0

    def test_conta_a_matriz_de_confusao(self) -> None:
        metrics = DetectionMetrics.of(
            predicted=[True, False, True, False], actual=[True, True, False, False]
        )
        assert (metrics.true_positives, metrics.false_positives) == (1, 1)
        assert (metrics.false_negatives, metrics.true_negatives) == (1, 1)


class TestGateDePromocao:
    BASELINE = RegressionMetrics(rmse=1.0, mae=0.8, r2=0.5)

    def test_melhora_expressiva_e_aprovada(self) -> None:
        challenger = RegressionMetrics(rmse=0.80, mae=0.6, r2=0.7)
        verdict = compare_against_baseline(
            challenger=challenger, baseline=self.BASELINE, min_improvement=0.05
        )
        assert verdict.approved is True
        assert verdict.improvement == pytest.approx(0.20)

    def test_empate_tecnico_e_reprovado(self) -> None:
        """Modelo complexo que empata com a persistência custa mais e não entrega."""
        challenger = RegressionMetrics(rmse=0.99, mae=0.79, r2=0.51)
        verdict = compare_against_baseline(
            challenger=challenger, baseline=self.BASELINE, min_improvement=0.05
        )
        assert verdict.approved is False
        assert "margem mínima" in verdict.reason

    def test_modelo_pior_e_reprovado(self) -> None:
        challenger = RegressionMetrics(rmse=1.5, mae=1.2, r2=0.2)
        verdict = compare_against_baseline(
            challenger=challenger, baseline=self.BASELINE, min_improvement=0.05
        )
        assert verdict.approved is False
        assert verdict.improvement < 0

    def test_exatamente_na_margem_e_aprovado(self) -> None:
        challenger = RegressionMetrics(rmse=0.95, mae=0.7, r2=0.6)
        verdict = compare_against_baseline(
            challenger=challenger, baseline=self.BASELINE, min_improvement=0.05
        )
        assert verdict.approved is True

    def test_veredito_registra_as_duas_metricas(self) -> None:
        """Auditabilidade: dá para reabrir a decisão meses depois."""
        challenger = RegressionMetrics(rmse=0.5, mae=0.4, r2=0.9)
        verdict = compare_against_baseline(
            challenger=challenger, baseline=self.BASELINE, min_improvement=0.05
        )
        assert verdict.challenger == challenger
        assert verdict.baseline == self.BASELINE

    def test_rejeita_margem_negativa(self) -> None:
        with pytest.raises(InvalidValueError, match="margem"):
            compare_against_baseline(
                challenger=self.BASELINE, baseline=self.BASELINE, min_improvement=-0.1
            )

    def test_baseline_perfeito_nao_divide_por_zero(self) -> None:
        perfect = RegressionMetrics(rmse=0.0, mae=0.0, r2=1.0)
        verdict = compare_against_baseline(
            challenger=RegressionMetrics(rmse=0.1, mae=0.1, r2=0.9),
            baseline=perfect,
            min_improvement=0.05,
        )
        assert verdict.approved is False
