"""LSTM Autoencoder em PyTorch e o adaptador que o liga ao domínio.

Requer o extra `[ml]`. Importar `eolica.infrastructure.ml` **não** puxa este
módulo — torch só entra no processo de quem vai treinar ou servir o modelo real.

## Diferença em relação ao autoencoder da v1

A v1 gerava a reconstrução de forma autorregressiva:

```python
decoder_input = x[:, -1, :].unsqueeze(1)
for t in range(sequence_length):
    output, hidden, cell = self.decoder(decoder_input, hidden, cell)
    outputs[:, t, :] = output.squeeze(1)
    decoder_input = output          # comentado como "teacher forcing"
```

Dois problemas:

1. **Aquilo não é teacher forcing.** Teacher forcing alimenta o *ground truth*
   do passo anterior; realimentar a própria saída é geração livre. O comentário
   descrevia o oposto do que o código fazia — e a distinção importa, porque
   geração livre acumula erro ao longo da sequência e torna o erro de
   reconstrução dependente do *comprimento* da janela, não só do seu conteúdo.
2. **Reconstruir para frente a partir do último passo** faz o modelo aprender a
   continuar a série, não a resumi-la. Para detecção de anomalia por
   reconstrução, o que se quer é o gargalo: comprimir a janela num vetor latente
   e reconstruí-la inteira a partir dele.

Aqui o latente é repetido ao longo dos passos (o padrão "RepeatVector") e
decodificado de uma vez. É determinístico, paraleliza no tempo e treina em uma
fração do tempo.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn

from eolica.domain.health import ReconstructionError
from eolica.domain.turbine import FEATURE_NAMES
from eolica.shared.errors import ConfigurationError

if TYPE_CHECKING:
    from eolica.domain.turbine import ReadingWindow


class LSTMAutoencoder(nn.Module):
    """Encoder-decoder LSTM para reconstrução de janelas multivariadas."""

    def __init__(
        self,
        *,
        n_features: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ConfigurationError("n_layers deve ser pelo menos 1", n_layers=n_layers)

        # O PyTorch ignora dropout com uma única camada e emite warning; zerar
        # explicitamente deixa a intenção clara e mantém o log limpo.
        effective_dropout = dropout if n_layers > 1 else 0.0

        self.n_features = n_features
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstrói `x`, de shape (batch, passos, features)."""
        _, (hidden, _) = self.encoder(x)
        latent = hidden[-1]  # (batch, hidden) — o gargalo
        repeated = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)  # type: ignore[no-any-return]

    @torch.no_grad()
    def window_errors(self, x: torch.Tensor) -> torch.Tensor:
        """MSE por amostra do batch."""
        return torch.mean((x - self(x)) ** 2, dim=(1, 2))


class StandardScaler:
    """Padronização por feature, serializável sem pickle.

    A v1 serializava o `StandardScaler` do scikit-learn com `pickle` e o
    registrava como artefato. Isso amarra o artefato à versão exata do sklearn e
    faz o carregamento executar código arbitrário do arquivo — um formato de
    troca ruim para algo que são dois vetores de floats.
    """

    def __init__(self, means: Sequence[float], deviations: Sequence[float]) -> None:
        self.means = tuple(float(m) for m in means)
        # Feature constante (sensor travado) não pode virar divisão por zero.
        self.deviations = tuple(float(d) if d > 0 else 1.0 for d in deviations)

    @classmethod
    def fit(cls, matrix: torch.Tensor) -> StandardScaler:
        """Ajusta sobre um tensor (amostras, features)."""
        return cls(
            means=matrix.mean(dim=0).tolist(),
            deviations=matrix.std(dim=0, unbiased=False).tolist(),
        )

    def transform(self, matrix: torch.Tensor) -> torch.Tensor:
        means = torch.tensor(self.means, dtype=matrix.dtype, device=matrix.device)
        deviations = torch.tensor(self.deviations, dtype=matrix.dtype, device=matrix.device)
        return (matrix - means) / deviations

    def to_dict(self) -> dict[str, list[float]]:
        """Representação JSON-serializável, para gravar como artefato."""
        return {"means": list(self.means), "deviations": list(self.deviations)}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> StandardScaler:
        return cls(means=payload["means"], deviations=payload["deviations"])


class TorchReconstructionModel:
    """Adaptador: satisfaz `domain.health.ports.ReconstructionModel`.

    Não herda do `Protocol` — a conformidade é estrutural, e a dependência
    aponta só para dentro.
    """

    def __init__(
        self,
        *,
        model: LSTMAutoencoder,
        scaler: StandardScaler,
        window_size: int,
        feature_names: Sequence[str] = FEATURE_NAMES,
        version: str = "lstm-autoencoder@local",
        batch_size: int = 256,
    ) -> None:
        if len(feature_names) != model.n_features:
            raise ConfigurationError(
                "O modelo espera um número de features diferente do declarado",
                model_features=model.n_features,
                declared=len(feature_names),
            )
        self._model = model.eval()
        self._scaler = scaler
        self._window_size = window_size
        self._feature_names = tuple(feature_names)
        self._version = version
        self._batch_size = batch_size

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def version(self) -> str:
        return self._version

    def reconstruction_errors(self, window: ReadingWindow) -> list[ReconstructionError]:
        """Erro de reconstrução de cada sub-janela deslizante.

        As sub-janelas são materializadas com `Tensor.unfold`, que devolve uma
        *view* sem copiar dados — relevante porque uma janela de um dia produz
        85 sub-janelas de 60×5, e copiar cada uma seria desperdício puro.
        """
        matrix = torch.tensor(window.matrix(self._feature_names), dtype=torch.float32)
        if matrix.size(0) < self._window_size:
            return []

        scaled = self._scaler.transform(matrix)
        # (n_windows, features, window) -> (n_windows, window, features)
        windows = scaled.unfold(dimension=0, size=self._window_size, step=1).transpose(1, 2)

        errors: list[float] = []
        for start in range(0, windows.size(0), self._batch_size):
            batch = windows[start : start + self._batch_size].contiguous()
            errors.extend(self._model.window_errors(batch).tolist())

        return [ReconstructionError(max(0.0, value)) for value in errors]
