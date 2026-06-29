"""Persistência: acesso à telemetria e validação do contrato de dado."""

from eolica.infrastructure.persistence.csv_repository import (
    DEFAULT_SAMPLING_INTERVAL,
    CsvScadaRepository,
)
from eolica.infrastructure.persistence.schemas import (
    COLUMN_MAPPING,
    EXCLUDED_CHANNELS,
    REQUIRED_COLUMNS,
    ScadaRecord,
    validate_scada_frame,
)

__all__ = [
    "COLUMN_MAPPING",
    "DEFAULT_SAMPLING_INTERVAL",
    "EXCLUDED_CHANNELS",
    "REQUIRED_COLUMNS",
    "CsvScadaRepository",
    "ScadaRecord",
    "validate_scada_frame",
]
