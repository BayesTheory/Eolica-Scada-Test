"""Features de janela móvel: dispersão e taxa de variação.

Vêm da análise exploratória da v1, que rankeava `*_std_1h` e `*_roc_1h` entre as
features mais preditivas para pré-falha — conclusão que nunca chegou ao código
porque o script apontava para um CSV que o pipeline não gerava.

A intuição física é boa e vale registrar: uma turbina prestes a falhar
raramente muda o *nível* das grandezas antes de mudar a *variabilidade* delas. A
temperatura média do gerador continua nos 40 °C enquanto o desvio padrão dentro
da hora começa a subir. É esse sinal que estas features capturam.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from eolica.infrastructure.featurestore.base import require_columns
from eolica.shared.errors import ConfigurationError, InsufficientDataError

SIGNATURE_PREFIX = "rollingview.v1"


@dataclass(frozen=True, slots=True)
class RollingFeatureView:
    """Desvio padrão e taxa de variação numa janela móvel — **causais**.

    O risco desta classe de feature é vazamento temporal. `rolling(6).std()`
    aplicado à série crua inclui o instante atual na janela, e o instante atual é
    exatamente o que se quer prever. Um modelo treinado assim exibe métrica
    offline excelente e não serve para nada em produção: na hora da decisão,
    aquele valor ainda não foi medido.

    Aqui a série é deslocada um passo **antes** de qualquer agregação. Toda
    feature para o alvo em `t` usa apenas observações até `t-1`, e o teste
    `test_alterar_o_futuro_nao_muda_nenhuma_feature_do_passado` verifica isso
    reescrevendo o futuro e exigindo que a matriz do passado não mude.
    """

    features: tuple[str, ...]
    target: str
    window_steps: int
    _feature_names: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.features:
            raise ConfigurationError("A lista de features não pode ser vazia")
        if self.window_steps < 2:
            raise ConfigurationError(
                "window_steps deve ser pelo menos 2 para haver dispersão",
                window_steps=self.window_steps,
            )
        if len(set(self.features)) != len(self.features):
            raise ConfigurationError("A lista de features tem duplicatas", features=self.features)

        names = tuple(
            f"{column}_{statistic}_{self.window_steps}"
            for column in sorted(self.features)
            for statistic in ("roc", "std")
        )
        object.__setattr__(self, "_feature_names", names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def source_columns(self) -> tuple[str, ...]:
        return tuple(sorted(self.features))

    @property
    def required_history(self) -> int:
        """Janela + 1.

        A taxa de variação compara o começo e o fim da janela, então precisa de
        um ponto a mais que o desvio padrão sozinho.
        """
        return self.window_steps + 1

    @property
    def signature(self) -> str:
        canonical = f"{sorted(self.features)}|{self.target}|{self.window_steps}"
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        return f"{SIGNATURE_PREFIX}:{digest}"

    def build_training_matrix(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Matriz histórica (X, y), sem nenhuma linha incompleta."""
        contract = f"RollingFeatureView({self.signature})"
        require_columns(frame, set(self.features) | {self.target}, contract=contract)

        columns: dict[str, pd.Series] = {}
        for column in sorted(self.features):
            # shift=1: para o alvo em t, a janela termina em t-1.
            columns.update(_rolling_columns(frame, column, self.window_steps, shift=1))

        rolled = pd.DataFrame(columns, index=frame.index)[list(self.feature_names)]
        complete = rolled.notna().all(axis=1)
        return rolled.loc[complete], frame.loc[complete, self.target]

    def build_inference_vector(self, history: pd.DataFrame) -> pd.DataFrame:
        """Vetor de uma linha para o instante seguinte ao fim de `history`."""
        require_columns(
            history, set(self.features), contract=f"RollingFeatureView({self.signature})"
        )
        if len(history) < self.required_history:
            raise InsufficientDataError(
                required=self.required_history, available=len(history), subject="observações"
            )

        row: dict[str, float] = {}
        for column in sorted(self.features):
            # shift=0: aqui o fim do histórico já *é* t-1, então não se desloca
            # de novo. Mesma função, mesmo cálculo — é o que impede o skew.
            computed = _rolling_columns(history, column, self.window_steps, shift=0)
            row.update({name: series.iloc[-1] for name, series in computed.items()})

        return pd.DataFrame([row], columns=list(self.feature_names), index=[history.index[-1]])


def _rolling_columns(
    frame: pd.DataFrame, column: str, window: int, *, shift: int
) -> dict[str, pd.Series]:
    """As duas estatísticas de janela de uma coluna. **Única implementação.**

    `shift=1` produz a série inteira alinhada por instante alvo (treino);
    `shift=0` produz a série cujo último elemento é o vetor de inferência.

    O desvio usa `ddof=0` (populacional) para casar com `numpy.std` e com o
    `statistics.pstdev` usado no domínio — três lugares calculando dispersão com
    convenções diferentes seria mais uma fonte silenciosa de divergência.
    """
    base = frame[column].shift(shift)
    return {
        f"{column}_roc_{window}": (base - base.shift(window)) / window,
        f"{column}_std_{window}": base.rolling(window).std(ddof=0),
    }
