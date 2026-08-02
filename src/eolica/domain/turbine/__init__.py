"""Subdomínio `turbine`: o ativo físico e sua telemetria.

Shared kernel entre os contextos de saúde e de previsão — os dois falam de
leituras da mesma turbina, mas decidem coisas diferentes sobre elas.
"""

from eolica.domain.turbine.entities import (
    FEATURE_NAMES,
    ReadingWindow,
    TurbineReading,
)
from eolica.domain.turbine.regimes import OperatingRegime
from eolica.domain.turbine.value_objects import (
    OperatingStatus,
    PitchAngle,
    PowerKw,
    RotorSpeed,
    Temperature,
    TurbineSpec,
    WindSpeed,
)

__all__ = [
    "FEATURE_NAMES",
    "OperatingRegime",
    "OperatingStatus",
    "PitchAngle",
    "PowerKw",
    "ReadingWindow",
    "RotorSpeed",
    "Temperature",
    "TurbineReading",
    "TurbineSpec",
    "WindSpeed",
]
