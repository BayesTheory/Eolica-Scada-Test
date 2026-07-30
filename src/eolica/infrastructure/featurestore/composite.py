"""Composição de feature views.

Um modelo raramente usa uma família de features só. Compor em vez de criar uma
classe `LagAndRollingFeatureView` mantém cada estatística testável isoladamente
e faz a garantia de ausência de skew valer para o conjunto sem novo esforço:
se cada parte não tem skew, a concatenação também não.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from eolica.infrastructure.featurestore.base import FeatureView
from eolica.shared.errors import ConfigurationError

SIGNATURE_PREFIX = "compositeview.v1"


@dataclass(frozen=True, slots=True)
class CompositeFeatureView:
    """Une várias views num único conjunto ordenado de features."""

    views: tuple[FeatureView, ...]
    target: str
    _feature_names: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.views:
            raise ConfigurationError("Uma view composta precisa de ao menos uma view")

        divergent = [view.target for view in self.views if view.target != self.target]
        if divergent:
            raise ConfigurationError(
                "Todas as views compostas devem prever o mesmo alvo",
                expected=self.target,
                found=sorted(set(divergent)),
            )

        names: list[str] = []
        for view in self.views:
            names.extend(view.feature_names)
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ConfigurationError(
                "Views compostas produzem colunas com o mesmo nome", duplicates=duplicates
            )

        object.__setattr__(self, "_feature_names", tuple(sorted(names)))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def source_columns(self) -> tuple[str, ...]:
        """União das colunas exigidas pelas views componentes."""
        return tuple(sorted({column for view in self.views for column in view.source_columns}))

    @property
    def required_history(self) -> int:
        """A exigência da view mais faminta.

        Uma linha só entra na matriz quando *todas* as views conseguem produzir
        suas colunas para ela.
        """
        return max(view.required_history for view in self.views)

    @property
    def signature(self) -> str:
        canonical = "|".join(sorted(view.signature for view in self.views)) + f"|{self.target}"
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        return f"{SIGNATURE_PREFIX}:{digest}"

    def build_training_matrix(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Concatena as matrizes das views, mantendo só instantes completos.

        O `join="inner"` é o que garante isso: um alvo cujo lag existe mas cuja
        janela de uma hora ainda não fechou fica de fora, em vez de entrar com
        `NaN` numa coluna.
        """
        matrices = [view.build_training_matrix(frame)[0] for view in self.views]
        combined = pd.concat(matrices, axis=1, join="inner")[list(self.feature_names)]
        return combined, frame.loc[combined.index, self.target]

    def build_inference_vector(self, history: pd.DataFrame) -> pd.DataFrame:
        vectors = [view.build_inference_vector(history) for view in self.views]
        combined = pd.concat(vectors, axis=1)[list(self.feature_names)]
        return combined
