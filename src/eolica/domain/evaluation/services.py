"""A regra que decide se um modelo vai para produção.

É regra de negócio, não detalhe de treino: define o que a organização considera
melhoria suficiente para justificar operar um modelo mais caro. Por isso vive no
domínio, com teste, e não dentro de um script de treino.
"""

from __future__ import annotations

from eolica.domain.evaluation.value_objects import PromotionVerdict, RegressionMetrics
from eolica.shared.errors import InvalidValueError


def compare_against_baseline(
    *,
    challenger: RegressionMetrics,
    baseline: RegressionMetrics,
    min_improvement: float,
) -> PromotionVerdict:
    """Decide se o desafiante supera o baseline por margem suficiente.

    A margem existe porque complexidade tem custo operacional. Um LSTM exige
    GPU, versionamento de pesos, monitoramento de drift e alguém que saiba
    depurá-lo às três da manhã. Se ele empata com uma média móvel de seis
    passos, a decisão certa é servir a média móvel.

    O v1 chamava `mlflow.register_model()` logo após o treino, incondicionalmente
    e sem comparação com nada — de modo que "o modelo foi para produção" não
    carregava nenhuma informação sobre ele ser bom.

    Args:
        challenger: métricas do modelo candidato.
        baseline: métricas do baseline (persistência, média móvel, z-score).
        min_improvement: redução relativa mínima de RMSE, em [0, 1].
            0.05 significa "precisa ser ao menos 5% melhor".
    """
    if min_improvement < 0:
        raise InvalidValueError(
            "A margem mínima de melhoria não pode ser negativa",
            min_improvement=min_improvement,
        )

    if baseline.rmse == 0:
        # Baseline perfeito: não há espaço para melhoria relativa, e qualquer
        # erro do desafiante é uma piora.
        improvement = 0.0 if challenger.rmse == 0 else float("-inf")
    else:
        improvement = (baseline.rmse - challenger.rmse) / baseline.rmse

    approved = improvement >= min_improvement

    if approved:
        reason = (
            f"RMSE {improvement:.1%} melhor que o baseline "
            f"({challenger.rmse:.4f} contra {baseline.rmse:.4f})."
        )
    elif improvement < 0:
        reason = (
            f"RMSE {abs(improvement):.1%} PIOR que o baseline "
            f"({challenger.rmse:.4f} contra {baseline.rmse:.4f}). Não promover."
        )
    else:
        reason = (
            f"Melhoria de {improvement:.1%} não atinge a margem mínima de "
            f"{min_improvement:.1%}: a complexidade extra não se justifica."
        )

    return PromotionVerdict(
        approved=approved,
        improvement=improvement,
        challenger=challenger,
        baseline=baseline,
        min_improvement=min_improvement,
        reason=reason,
    )
