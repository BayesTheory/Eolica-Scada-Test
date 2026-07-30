"""Contrato comum das feature views.

Uma feature view sabe três coisas: quais colunas produz (e em que ordem), quanto
histórico exige, e como se identificar. As duas rotas de materialização —
treino e serving — são obrigação de quem implementa, e a garantia do projeto é
que ambas passem pela mesma função privada.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from eolica.shared.errors import ContractViolationError

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class FeatureView(Protocol):
    """Um conjunto de features derivadas, materializável de duas formas."""

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Colunas produzidas, na ordem exata que o modelo deve receber."""
        ...

    @property
    def required_history(self) -> int:
        """Observações mínimas para montar um vetor de inferência."""
        ...

    @property
    def signature(self) -> str:
        """Identidade estável, gravada junto do modelo treinado."""
        ...

    @property
    def target(self) -> str: ...

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Colunas cruas de que a view depende.

        Distinto de `feature_names`: estas são as colunas de entrada (`power`,
        `wind_speed`), aquelas são as derivadas (`power_lag_1`, `power_std_6`).
        Quem monta o DataFrame de inferência precisa saber quais buscar.
        """
        ...

    def build_training_matrix(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: ...

    def build_inference_vector(self, history: pd.DataFrame) -> pd.DataFrame: ...


def require_columns(frame: pd.DataFrame, needed: set[str], *, contract: str) -> None:
    """Exige as colunas declaradas, listando todas as ausentes de uma vez."""
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ContractViolationError(
            contract=contract,
            violations=[f"coluna ausente: '{name}'" for name in missing],
        )
