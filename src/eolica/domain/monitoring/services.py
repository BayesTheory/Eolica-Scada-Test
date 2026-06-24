"""Estatísticas de drift, em Python puro.

Tanto PSI quanto KS existem prontos em `scipy`/`evidently`. São reimplementados
aqui por uma razão arquitetural: o domínio não importa terceiros. O custo é ~60
linhas testadas contra valores conhecidos; o ganho é que a regra de "quando
retreinar" não fica refém da API de uma biblioteca.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence

from eolica.domain.monitoring.value_objects import DriftMethod, DriftScore
from eolica.shared.errors import InsufficientDataError, InvalidValueError

# Massa mínima atribuída a um bin vazio, para o log não estourar.
_EPSILON = 1e-6


def _require_samples(reference: Sequence[float], current: Sequence[float]) -> None:
    if not reference:
        raise InsufficientDataError(required=1, available=0, subject="amostras de referência")
    if not current:
        raise InsufficientDataError(required=1, available=0, subject="amostras atuais")


def _quantile_edges(values: Sequence[float], bins: int) -> list[float]:
    """Limites de bin por equifrequência sobre a amostra de referência.

    Equifrequência (e não largura fixa) porque as features do SCADA são muito
    assimétricas: `PowerOutput` tem mediana 0.005 kW e máximo 6.9 kW. Bins de
    largura igual jogariam 90% das amostras no primeiro bin e o PSI perderia
    resolução justamente onde há massa.
    """
    ordered = sorted(values)
    size = len(ordered)
    edges: list[float] = []
    for index in range(1, bins):
        rank = (size - 1) * (index / bins)
        low = math.floor(rank)
        high = math.ceil(rank)
        edge = (
            ordered[low]
            if low == high
            else ordered[low] + (rank - low) * (ordered[high] - ordered[low])
        )
        edges.append(edge)
    return edges


def _proportions(values: Sequence[float], edges: Sequence[float]) -> list[float]:
    """Fração da amostra em cada bin definido por `edges`."""
    counts = [0] * (len(edges) + 1)
    for value in values:
        counts[bisect.bisect_right(edges, value)] += 1
    total = len(values)
    return [count / total for count in counts]


def population_stability_index(
    *, reference: Sequence[float], current: Sequence[float], bins: int = 10
) -> DriftScore:
    """PSI entre a distribuição de referência (treino) e a atual (produção).

        PSI = Σ (pₐ − pᵣ) · ln(pₐ / pᵣ)

    Bins vazios recebem `_EPSILON` — sem isso, uma única feature que saiu do
    suporte de treino produz `log(0)` e derruba o job de monitoramento em vez de
    reportar o drift gritante que acabou de detectar.
    """
    if bins < 2:
        raise InvalidValueError("São necessários pelo menos 2 bins", bins=bins)
    _require_samples(reference, current)

    edges = (
        _degenerate_edges(reference)
        if _is_constant(reference)
        else _quantile_edges(reference, bins)
    )

    reference_pct = _proportions(reference, edges)
    current_pct = _proportions(current, edges)

    total = 0.0
    for expected, actual in zip(reference_pct, current_pct, strict=True):
        safe_expected = max(expected, _EPSILON)
        safe_actual = max(actual, _EPSILON)
        total += (safe_actual - safe_expected) * math.log(safe_actual / safe_expected)

    return DriftScore(value=total, method=DriftMethod.PSI)


def _is_constant(values: Sequence[float]) -> bool:
    first = values[0]
    return all(value == first for value in values)


def _degenerate_edges(values: Sequence[float]) -> list[float]:
    """Bins para uma referência sem variância alguma.

    Acontece de verdade: `PitchDeg` fica travado em 14.034 por períodos longos,
    e um sensor congelado produz exatamente isso. Os quantis colapsariam num
    único ponto e todo valor novo cairia no mesmo bin — o PSI daria zero
    justamente quando a feature começasse a variar. Com três bins
    (abaixo | igual | acima) a mudança fica visível.
    """
    constant = values[0]
    return [math.nextafter(constant, -math.inf), constant]


def kolmogorov_smirnov(*, reference: Sequence[float], current: Sequence[float]) -> DriftScore:
    """Estatística D de Kolmogorov–Smirnov para duas amostras.

    D é a maior distância vertical entre as duas funções de distribuição
    acumulada empíricas — 0 para amostras idênticas, 1 para suportes disjuntos.

    Complementa o PSI: o PSI é sensível a mudança de *massa* entre faixas, o KS
    a mudança de *forma*. Uma distribuição que fica bimodal mantendo a média
    engana o PSI e não engana o KS.
    """
    _require_samples(reference, current)

    ordered_reference = sorted(reference)
    ordered_current = sorted(current)
    reference_size = len(ordered_reference)
    current_size = len(ordered_current)

    largest_gap = 0.0
    for value in set(ordered_reference) | set(ordered_current):
        cdf_reference = bisect.bisect_right(ordered_reference, value) / reference_size
        cdf_current = bisect.bisect_right(ordered_current, value) / current_size
        largest_gap = max(largest_gap, abs(cdf_reference - cdf_current))

    return DriftScore(value=largest_gap, method=DriftMethod.KS)
