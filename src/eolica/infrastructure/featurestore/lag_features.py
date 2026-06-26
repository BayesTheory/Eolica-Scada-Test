"""Feature view de lags: a única implementação de features tabulares defasadas.

Regra deste módulo: **treino e serving compartilham `_lag_column`.** Não há uma
segunda forma de construir um lag em lugar nenhum do projeto, e o teste
`TestAusenciaDeSkew` falha se alguém criar uma.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from eolica.shared.errors import (
    ConfigurationError,
    ContractViolationError,
    InsufficientDataError,
)

SIGNATURE_PREFIX = "lagview.v1"


@dataclass(frozen=True, slots=True)
class LagFeatureView:
    """Define — e é a única a materializar — um conjunto de features de lag.

    A ordem das colunas é derivada, nunca informada: `sorted(features)` × lag
    crescente. Isso remove a possibilidade de treino e serving concordarem
    sobre *quais* features usar e discordarem sobre a *ordem* delas, que num
    modelo baseado em árvore não levanta erro nenhum — só piora a previsão.
    """

    features: tuple[str, ...]
    target: str
    n_lags: int
    _feature_names: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.features:
            raise ConfigurationError("A lista de features não pode ser vazia")
        if self.n_lags < 1:
            raise ConfigurationError("n_lags deve ser pelo menos 1", n_lags=self.n_lags)
        if len(set(self.features)) != len(self.features):
            raise ConfigurationError("A lista de features tem duplicatas", features=self.features)

        names = tuple(
            f"{column}_lag_{lag}"
            for column in sorted(self.features)
            for lag in range(1, self.n_lags + 1)
        )
        object.__setattr__(self, "_feature_names", names)

    # ── contrato ─────────────────────────────────────────────────────────────

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Nomes das colunas, na ordem exata que o modelo deve receber."""
        return self._feature_names

    @property
    def required_history(self) -> int:
        """Observações mínimas para montar um vetor de inferência."""
        return self.n_lags

    @property
    def signature(self) -> str:
        """Identidade estável desta view, para gravar junto do modelo treinado.

        Serve para transformar o bug silencioso do v1 em erro alto: o adaptador
        do registry compara a assinatura gravada no treino com a da view em uso,
        e recusa servir se divergirem. Mudar `n_lags` de 6 para 12 passa a
        derrubar o readiness probe em vez de degradar a previsão em silêncio.
        """
        canonical = f"{sorted(self.features)}|{self.target}|{self.n_lags}"
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        return f"{SIGNATURE_PREFIX}:{digest}"

    # ── materialização ───────────────────────────────────────────────────────

    def build_training_matrix(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Matriz histórica (X, y) para treino e backtest.

        Linhas sem histórico completo são descartadas — nunca imputadas. Um lag
        preenchido com a média é uma informação que não existia no instante da
        decisão, e treinar com ela infla a métrica offline sem melhorar nada em
        produção.
        """
        self._require_columns(frame, extra=(self.target,))

        lagged = pd.DataFrame(
            {
                name: _lag_column(frame, column, lag)
                for column in sorted(self.features)
                for lag, name in ((lag, f"{column}_lag_{lag}") for lag in range(1, self.n_lags + 1))
            },
            index=frame.index,
        )[list(self.feature_names)]

        complete = lagged.notna().all(axis=1)
        return lagged.loc[complete], frame.loc[complete, self.target]

    def build_inference_vector(self, history: pd.DataFrame) -> pd.DataFrame:
        """Vetor de uma linha para prever o instante seguinte ao fim de `history`.

        Equivalente, por construção, à linha que `build_training_matrix`
        produziria para esse mesmo instante alvo.
        """
        self._require_columns(history)

        if len(history) < self.required_history:
            raise InsufficientDataError(
                required=self.required_history, available=len(history), subject="observações"
            )

        row = {
            f"{column}_lag_{lag}": _lag_column(history, column, lag - 1).iloc[-1]
            for column in sorted(self.features)
            for lag in range(1, self.n_lags + 1)
        }
        return pd.DataFrame([row], columns=list(self.feature_names), index=[history.index[-1]])

    # ── validação ────────────────────────────────────────────────────────────

    def _require_columns(self, frame: pd.DataFrame, extra: tuple[str, ...] = ()) -> None:
        needed = set(self.features) | set(extra)
        missing = sorted(needed - set(frame.columns))
        if missing:
            raise ContractViolationError(
                contract=f"LagFeatureView({self.signature})",
                violations=[f"coluna ausente: '{name}'" for name in missing],
            )


def _lag_column(frame: pd.DataFrame, column: str, lag: int) -> pd.Series:
    """A defasagem de `lag` passos de uma coluna. **Única implementação.**

    `build_training_matrix` chama com `lag=k` para obter a série inteira
    deslocada; `build_inference_vector` chama com `lag=k-1` e pega o último
    elemento — que é o mesmo valor, já que ali o "instante alvo" é uma posição à
    frente do fim do histórico.
    """
    return frame[column].shift(lag)
