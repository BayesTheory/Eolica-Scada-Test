"""Feature store: onde uma feature é definida uma vez e usada em toda parte.

Não é um Feast nem um Tecton — é o mínimo que resolve o problema real deste
sistema: garantir que a feature vista no treino é bit a bit a mesma vista no
serving, e que o modelo carrega a identidade do conjunto com que foi treinado.
"""

from eolica.infrastructure.featurestore.lag_features import LagFeatureView

__all__ = ["LagFeatureView"]
