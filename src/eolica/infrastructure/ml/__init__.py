"""Adaptadores de modelos de machine learning.

`baselines` não tem dependência pesada e está sempre disponível. Os adaptadores
reais (`torch_autoencoder`, `xgboost_forecaster`, `mlflow_registry`) exigem o
extra `[ml]` e são importados sob demanda — importar este pacote nunca puxa
torch para dentro do processo.
"""

from eolica.infrastructure.ml.baselines import (
    MovingAverageForecaster,
    PersistenceForecaster,
    ZScoreBaselineDetector,
)

__all__ = ["MovingAverageForecaster", "PersistenceForecaster", "ZScoreBaselineDetector"]
