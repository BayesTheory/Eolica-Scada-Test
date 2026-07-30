"""Feature store: onde uma feature é definida uma vez e usada em toda parte.

Não é um Feast nem um Tecton — é o mínimo que resolve o problema real deste
sistema: garantir que a feature vista no treino é bit a bit a mesma vista no
serving, e que o modelo carrega a identidade do conjunto com que foi treinado.

Duas garantias valem para toda view aqui:

- **Sem skew.** Treino e serving passam pela mesma função privada; um teste
  compara as duas rotas para o mesmo instante alvo.
- **Causalidade.** Nenhuma feature para o alvo em `t` enxerga observação de `t`
  ou depois. Vale principalmente para as janelas móveis, onde `rolling().std()`
  ingênuo incluiria o próprio instante que se quer prever.
"""

from eolica.infrastructure.featurestore.base import FeatureView
from eolica.infrastructure.featurestore.composite import CompositeFeatureView
from eolica.infrastructure.featurestore.lag_features import LagFeatureView
from eolica.infrastructure.featurestore.rolling_features import RollingFeatureView

__all__ = [
    "CompositeFeatureView",
    "FeatureView",
    "LagFeatureView",
    "RollingFeatureView",
]
