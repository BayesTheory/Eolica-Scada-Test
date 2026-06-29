"""Contrato de dado da telemetria SCADA.

A fronteira onde um DataFrame vira domínio é o único lugar onde dado malformado
pode ser barrado de graça. Depois disso ele já está espalhado por três camadas.

O v1 não tinha essa fronteira: `pd.read_csv` no import do módulo, e a primeira
notícia de que uma coluna havia sumido era um `KeyError` dentro do laço de
inferência do PyTorch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

from eolica.shared.errors import ContractViolationError

if TYPE_CHECKING:
    import pandas as pd

# ── mapeamento coluna do CSV → nome canônico do domínio ──────────────────────
# À direita, o nome IEC 61400-25 documentado em data/metadata/scada_channels.csv.
COLUMN_MAPPING: Final[dict[str, str]] = {
    "WindSpeed": "wind_speed",  # WMET.HorWdSpd
    "PowerOutput": "power",  # WCNV.kW
    "RotorSpeed": "rotor_speed",  # WROT.RotSpd
    "GeneratorTemperature": "generator_temperature",  # WGEN.SttTmp
    "PitchDeg": "pitch",  # WROT.BlPthAngVal
    "Status_rounded": "status",  # SERVER.TurSt (arredondado)
}

REQUIRED_COLUMNS: Final[tuple[str, ...]] = tuple(COLUMN_MAPPING)
INDEX_NAME: Final[str] = "Datetime"

# `GeneratorSpeed` está deliberadamente fora. O metadado do fabricante marca o
# canal como `Reliable Measurement = FALSE` — "0-10V Signal from Generator (not
# an exact measurement)". O v1 o listava entre as features de treino.
EXCLUDED_CHANNELS: Final[frozenset[str]] = frozenset({"GeneratorSpeed"})


class ScadaRecord(BaseModel):
    """Uma observação SCADA validada.

    Os limites vêm do envelope físico da Aventa AV-7 e do que o dataset real
    exibe, com folga. São generosos de propósito: o objetivo é pegar sensor
    quebrado e erro de unidade (um vento de 900 m/s, uma temperatura de -400 °C),
    não fazer detecção de anomalia — isso é trabalho do modelo.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    wind_speed: float = Field(ge=0.0, le=100.0, description="m/s, anemômetro da nacele")
    power: float = Field(ge=-10.0, le=50.0, description="kW no conversor; negativo = parasita")
    rotor_speed: float = Field(ge=0.0, le=500.0, description="RPM")
    generator_temperature: float = Field(gt=-273.15, le=250.0, description="°C no estator")
    pitch: float = Field(ge=-180.0, le=180.0, description="graus")
    status: float = Field(description="Código de estado da turbina")


def validate_scada_frame(frame: pd.DataFrame, *, contract: str = "scada.raw.v1") -> None:
    """Valida um DataFrame contra o contrato, acumulando **todas** as violações.

    Acumular em vez de falhar na primeira é deliberado: quem está consertando um
    pipeline de ingestão quer a lista inteira do que está errado, não uma
    descoberta por execução.
    """
    violations: list[str] = []

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    violations.extend(f"coluna obrigatória ausente: '{name}'" for name in missing)

    if frame.empty:
        violations.append("o dataframe não tem nenhuma linha")

    if not missing and not frame.empty:
        violations.extend(_column_violations(frame))
        violations.extend(_index_violations(frame))

    if violations:
        raise ContractViolationError(contract=contract, violations=violations)


def _column_violations(frame: pd.DataFrame) -> list[str]:
    violations: list[str] = []
    for column in REQUIRED_COLUMNS:
        series = frame[column]
        null_count = int(series.isna().sum())
        if null_count:
            violations.append(f"'{column}' tem {null_count} valor(es) nulo(s)")
        if not _is_numeric(series):
            violations.append(f"'{column}' não é numérica (dtype={series.dtype})")
    return violations


def _index_violations(frame: pd.DataFrame) -> list[str]:
    import pandas as pd

    violations: list[str] = []
    if not isinstance(frame.index, pd.DatetimeIndex):
        violations.append(f"o índice deve ser DatetimeIndex, veio {type(frame.index).__name__}")
        return violations
    if not frame.index.is_monotonic_increasing:
        violations.append("o índice temporal não está ordenado")
    duplicates = int(frame.index.duplicated().sum())
    if duplicates:
        violations.append(f"o índice tem {duplicates} timestamp(s) duplicado(s)")
    return violations


def _is_numeric(series: pd.Series) -> bool:
    import pandas as pd

    return bool(pd.api.types.is_numeric_dtype(series))
