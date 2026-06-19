"""Portas do subdomínio `health`.

O domínio declara *o que* precisa de um modelo de reconstrução; a infraestrutura
decide se isso é um LSTM em PyTorch, um ONNX ou um stub. Como é `Protocol`, o
adaptador não herda nada daqui — a dependência aponta só para dentro.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eolica.domain.health.value_objects import ReconstructionError
    from eolica.domain.turbine import ReadingWindow


@runtime_checkable
class ReconstructionModel(Protocol):
    """Um modelo capaz de reconstruir janelas e reportar o erro."""

    @property
    def window_size(self) -> int:
        """Passos de tempo que o modelo espera por janela."""
        ...

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Features esperadas, **na ordem em que o modelo foi treinado**.

        Expor isto na porta é o que permite ao feature store verificar a ordem
        em vez de confiar nela. No v1 a ordem das colunas era garantida por
        coincidência entre duas listas escritas à mão em arquivos diferentes.
        """
        ...

    def reconstruction_errors(self, window: ReadingWindow) -> Sequence[ReconstructionError]:
        """Erro de reconstrução para cada sub-janela deslizante de `window`."""
        ...
