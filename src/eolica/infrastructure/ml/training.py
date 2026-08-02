"""Pipelines de treino, com o gate de promoção contra baseline.

Requer o extra `[ml]`.

A diferença estrutural em relação ao v1 é uma linha de código e uma mudança de
postura. O v1 fazia:

    mlflow.xgboost.log_model(best_model.model, artifact_path="model")
    mlflow.register_model(model_uri=model_uri, name=POWER_FORECASTER_MODEL_NAME)

Registrar era incondicional. Todo treino que não estourasse exceção virava um
modelo registrado, e como o alias servido era `latest`, virava o modelo de
produção. Não havia baseline, então "R² médio 0.87" não dizia se aquilo era bom.

Aqui o treino sempre loga métricas e artefatos — histórico de experimento é
valioso mesmo para o modelo que não presta — mas só **registra** quem passa pelo
gate de domínio `compare_against_baseline`. Promover para um alias servível
continua sendo ato humano separado.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eolica.domain.evaluation import (
    PromotionVerdict,
    RegressionMetrics,
    compare_against_baseline,
)
from eolica.domain.forecasting import Horizon
from eolica.domain.turbine import FEATURE_NAMES, ReadingWindow
from eolica.infrastructure.ml.baselines import MovingAverageForecaster
from eolica.infrastructure.ml.mlflow_registry import (
    TAG_BEATS_BASELINE,
    TAG_FEATURE_NAMES,
    TAG_FEATURE_SIGNATURE,
    TAG_WINDOW_SIZE,
)
from eolica.shared.errors import InsufficientDataError

if TYPE_CHECKING:
    import pandas as pd

    from eolica.infrastructure.featurestore import FeatureView

DEFAULT_MIN_IMPROVEMENT = 0.05
DEFAULT_VALIDATION_FRACTION = 0.2


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """O que o treino produziu — inclusive quando não é promovível."""

    verdict: PromotionVerdict
    registered: bool
    run_id: str | None
    model_version: str | None
    training_rows: int
    validation_rows: int

    @property
    def summary(self) -> str:
        status = "registrado" if self.registered else "NÃO registrado"
        return f"{status}: {self.verdict.reason}"


def _temporal_split(
    features: pd.DataFrame, target: pd.Series, *, validation_fraction: float
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Corte temporal, nunca aleatório.

    Embaralhar uma série temporal põe o futuro no conjunto de treino e produz
    métrica de validação que não significa nada. O v1 usava `TimeSeriesSplit`
    para o forecasting, o que estava correto — mas o modelo de anomalia era
    partido com um `int(len(df) * 0.8)` simples, sem gap entre os conjuntos.
    """
    split = int(len(features) * (1 - validation_fraction))
    if split < 1 or split >= len(features):
        raise InsufficientDataError(
            required=int(1 / validation_fraction) + 1,
            available=len(features),
            subject="amostras para partir treino e validação",
        )
    return (
        features.iloc[:split],
        target.iloc[:split],
        features.iloc[split:],
        target.iloc[split:],
    )


def train_forecast_model(
    *,
    frame: pd.DataFrame,
    view: FeatureView,
    registry_name: str,
    tracker: Any,
    params: dict[str, Any] | None = None,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> TrainingOutcome:
    """Treina o previsor de potência e o compara com a média móvel.

    O baseline é avaliado **no mesmo conjunto de validação**, com as mesmas
    linhas — comparar contra um baseline medido noutro recorte seria comparar
    com nada.
    """
    import mlflow
    import xgboost as xgb

    features, target = view.build_training_matrix(frame)
    x_train, y_train, x_valid, y_valid = _temporal_split(
        features, target, validation_fraction=validation_fraction
    )

    hyperparameters = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.9,
        "random_state": 0,
        **(params or {}),
    }

    with tracker.start_run(run_name=f"forecast_{view.signature}") as run:
        mlflow.log_params(hyperparameters)
        mlflow.log_param("feature_view", view.signature)
        mlflow.log_param("n_features", len(view.feature_names))

        booster = xgb.XGBRegressor(**hyperparameters)
        booster.fit(x_train, y_train)

        challenger = RegressionMetrics.of(
            actual=list(y_valid), predicted=[float(p) for p in booster.predict(x_valid)]
        )
        baseline = _persistence_baseline(frame, y_valid)

        verdict = compare_against_baseline(
            challenger=challenger, baseline=baseline, min_improvement=min_improvement
        )

        mlflow.log_metrics(
            {
                "rmse": challenger.rmse,
                "mae": challenger.mae,
                "r2": challenger.r2,
                "baseline_rmse": baseline.rmse,
                "baseline_mae": baseline.mae,
                "improvement_over_baseline": verdict.improvement,
            }
        )
        mlflow.set_tag(TAG_BEATS_BASELINE, str(verdict.approved).lower())
        mlflow.set_tag(TAG_FEATURE_SIGNATURE, view.signature)
        mlflow.xgboost.log_model(booster, name="model")

        version = None
        if verdict.approved:
            registered = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=registry_name,
                tags={
                    TAG_FEATURE_SIGNATURE: view.signature,
                    TAG_FEATURE_NAMES: json.dumps(list(view.feature_names)),
                    TAG_BEATS_BASELINE: "true",
                },
            )
            version = str(registered.version)

        return TrainingOutcome(
            verdict=verdict,
            registered=verdict.approved,
            run_id=run.info.run_id,
            model_version=version,
            training_rows=len(x_train),
            validation_rows=len(x_valid),
        )


def _persistence_baseline(frame: pd.DataFrame, y_valid: pd.Series) -> RegressionMetrics:
    """Previsão por persistência sobre as mesmas linhas de validação.

    ŷ(t) = y(t-1). É o baseline mais honesto para série temporal e o mais
    difícil de bater em horizonte curto — motivo pelo qual quase todo projeto de
    forecasting que não o mede acaba reportando ganho ilusório.
    """
    target_column = y_valid.name
    previous = frame[target_column].shift(1).loc[y_valid.index]
    return RegressionMetrics.of(
        actual=[float(v) for v in y_valid], predicted=[float(v) for v in previous]
    )


def train_health_model(
    *,
    windows: list[ReadingWindow],
    registry_name: str,
    tracker: Any,
    window_size: int,
    sampling_interval: timedelta,
    hidden_size: int = 128,
    n_layers: int = 2,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> TrainingOutcome:
    """Treina o autoencoder e o compara com o baseline z-score.

    A comparação usa erro de reconstrução médio no conjunto de validação — se o
    autoencoder não reconstrói operação normal melhor que uma padronização
    simples, ele não aprendeu a assinatura da máquina, só o ruído.
    """
    import mlflow
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from eolica.infrastructure.ml.baselines import ZScoreBaselineDetector
    from eolica.infrastructure.ml.torch_autoencoder import (
        LSTMAutoencoder,
        StandardScaler,
        TorchReconstructionModel,
    )

    if not windows:
        raise InsufficientDataError(required=1, available=0, subject="janelas de operação normal")

    split = max(1, int(len(windows) * (1 - validation_fraction)))
    train_windows, valid_windows = windows[:split], windows[split:] or windows[-1:]

    rows = [row for window in train_windows for row in window.matrix(FEATURE_NAMES)]
    matrix = torch.tensor(rows, dtype=torch.float32)
    scaler = StandardScaler.fit(matrix)

    sequences = _to_sequences(train_windows, scaler, window_size)
    validation = _to_sequences(valid_windows, scaler, window_size)

    with tracker.start_run(run_name=f"health_autoencoder_w{window_size}") as run:
        mlflow.log_params(
            {
                "hidden_size": hidden_size,
                "n_layers": n_layers,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "window_size": window_size,
                "n_features": len(FEATURE_NAMES),
            }
        )

        network = LSTMAutoencoder(
            n_features=len(FEATURE_NAMES), hidden_size=hidden_size, n_layers=n_layers
        )
        optimiser = torch.optim.AdamW(network.parameters(), lr=learning_rate)
        loader = DataLoader(TensorDataset(sequences), batch_size=batch_size, shuffle=True)

        best_validation = float("inf")
        best_state: dict[str, Any] | None = None
        for epoch in range(epochs):
            network.train()
            for (batch,) in loader:
                optimiser.zero_grad()
                loss = torch.nn.functional.mse_loss(network(batch), batch)
                # `Tensor.backward` não tem anotação nos stubs do torch.
                loss.backward()  # type: ignore[no-untyped-call]
                optimiser.step()

            network.eval()
            with torch.no_grad():
                validation_loss = float(
                    torch.nn.functional.mse_loss(network(validation), validation)
                )
            mlflow.log_metric("validation_mse", validation_loss, step=epoch)

            if validation_loss < best_validation:
                best_validation = validation_loss
                best_state = {k: v.clone() for k, v in network.state_dict().items()}

        if best_state is not None:
            network.load_state_dict(best_state)

        baseline_detector = ZScoreBaselineDetector.fit(train_windows, window_size=window_size)
        baseline_errors = [
            error.value
            for window in valid_windows
            for error in baseline_detector.reconstruction_errors(window)
        ]

        challenger = RegressionMetrics(rmse=best_validation**0.5, mae=best_validation, r2=0.0)
        baseline_mse = sum(baseline_errors) / len(baseline_errors) if baseline_errors else 1.0
        baseline = RegressionMetrics(rmse=baseline_mse**0.5, mae=baseline_mse, r2=0.0)
        verdict = compare_against_baseline(
            challenger=challenger, baseline=baseline, min_improvement=min_improvement
        )

        mlflow.log_metrics(
            {
                "reconstruction_mse": best_validation,
                "baseline_reconstruction_mse": baseline_mse,
                "improvement_over_baseline": verdict.improvement,
            }
        )
        mlflow.set_tag(TAG_BEATS_BASELINE, str(verdict.approved).lower())

        with tempfile.TemporaryDirectory() as directory:
            scaler_path = Path(directory) / "scaler.json"
            scaler_path.write_text(json.dumps(scaler.to_dict()), encoding="utf-8")
            mlflow.log_artifact(str(scaler_path))

        mlflow.pytorch.log_model(network, name="model")

        version = None
        if verdict.approved:
            registered = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=registry_name,
                tags={
                    TAG_WINDOW_SIZE: str(window_size),
                    TAG_FEATURE_NAMES: json.dumps(list(FEATURE_NAMES)),
                    TAG_BEATS_BASELINE: "true",
                },
            )
            version = str(registered.version)

        # Construído para validar que o artefato treinado é servível de fato,
        # antes de qualquer promoção.
        TorchReconstructionModel(
            model=network, scaler=scaler, window_size=window_size, feature_names=FEATURE_NAMES
        )

        return TrainingOutcome(
            verdict=verdict,
            registered=verdict.approved,
            run_id=run.info.run_id,
            model_version=version,
            training_rows=int(sequences.size(0)),
            validation_rows=int(validation.size(0)),
        )


def _to_sequences(windows: list[ReadingWindow], scaler: Any, window_size: int) -> Any:
    """Empilha as sub-janelas deslizantes de vários segmentos num tensor."""
    import torch

    chunks = []
    for window in windows:
        matrix = torch.tensor(window.matrix(FEATURE_NAMES), dtype=torch.float32)
        if matrix.size(0) < window_size:
            continue
        scaled = scaler.transform(matrix)
        chunks.append(scaled.unfold(dimension=0, size=window_size, step=1).transpose(1, 2))

    if not chunks:
        raise InsufficientDataError(required=window_size, available=0, subject="passos contíguos")
    return torch.cat(chunks, dim=0).contiguous()


__all__ = [
    "Horizon",
    "MovingAverageForecaster",
    "TrainingOutcome",
    "train_forecast_model",
    "train_health_model",
]
