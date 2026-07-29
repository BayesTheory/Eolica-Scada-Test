"""Adaptadores de modelos de machine learning.

`baselines` não tem dependência pesada e está sempre disponível. Os adaptadores
reais (`torch_autoencoder`, `xgboost_forecaster`) exigem o extra `[ml]` e são
importados sob demanda — importar este pacote nunca puxa torch para dentro do
processo.

Ainda não implementado: o adaptador de carregamento a partir do MLflow Registry.
Os adaptadores atuais recebem modelo e scaler já construídos, o que basta para
treino local e para os testes. O `Settings` já carrega `mlflow_tracking_uri`,
`model_stage` e os nomes registrados; falta o módulo que os usa para resolver
uma versão promovida e conferir a assinatura da feature view antes de servir.
"""

from eolica.infrastructure.ml.baselines import (
    MovingAverageForecaster,
    PersistenceForecaster,
    ZScoreBaselineDetector,
)

__all__ = ["MovingAverageForecaster", "PersistenceForecaster", "ZScoreBaselineDetector"]
