"""Pipeline de ingestão: SCADA bruto a 1 Hz → grade de 10 minutos validada.

Substitui `pipeline_data.py`, cujo ponto de entrada estava quebrado: `main.py`
chamava `pipeline_data.main()`, função que não existia no módulo. O comando
`python main.py process_data` documentado no README morria com `AttributeError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from eolica.infrastructure.persistence.schemas import validate_scada_frame
from eolica.shared.errors import DataSourceError

# Amostras esperadas numa janela de 10 minutos a 1 Hz.
EXPECTED_SAMPLES_PER_WINDOW = 600
QUALITY_RATIO = 0.9

PRODUCING_STATUS = 10


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """O que a ingestão produziu — para log e para o relatório da CLI."""

    raw_rows: int
    resampled_rows: int
    rejected_by_quality: int
    normal_operation_rows: int
    output_path: Path

    @property
    def quality_rejection_ratio(self) -> float:
        total = self.resampled_rows + self.rejected_by_quality
        return 0.0 if total == 0 else self.rejected_by_quality / total


def ingest_scada(
    *,
    raw_path: Path,
    output_path: Path,
    interval: str = "10min",
) -> IngestionResult:
    """Reamostra, filtra por qualidade e valida contra o contrato.

    O filtro de qualidade exige 90% das 600 amostras esperadas em cada janela de
    10 minutos. Uma janela com 50 leituras tem média tão instável que alimentar
    o autoencoder com ela produz erro de reconstrução alto — indistinguível de
    anomalia real da turbina.
    """
    if not raw_path.exists():
        raise DataSourceError(
            f"Arquivo SCADA bruto não encontrado: {raw_path}. "
            "Baixe o dataset de https://zenodo.org/records/15700928",
            path=str(raw_path),
        )

    raw = pd.read_csv(raw_path)
    if "Datetime" not in raw.columns:
        raise DataSourceError("O CSV bruto não tem a coluna 'Datetime'", path=str(raw_path))

    raw["Datetime"] = pd.to_datetime(raw["Datetime"])
    raw = raw.set_index("Datetime").sort_index()

    numeric = raw.select_dtypes(include="number")
    resampled = numeric.resample(interval)

    # Duas agregações separadas em vez de um `.agg({col: [...]})` misto. A versão
    # com dicionário produz colunas em MultiIndex que precisam ser achatadas com
    # `"_".join(...)` e depois renomeadas para remover o sufixo `_mean` — três
    # passos frágeis onde bastam dois explícitos.
    aggregated = resampled.mean()
    # Amostras efetivamente presentes na janela: alimenta o filtro de qualidade.
    aggregated["WindSpeed_count"] = numeric["WindSpeed"].resample(interval).count()

    threshold = QUALITY_RATIO * EXPECTED_SAMPLES_PER_WINDOW
    accepted = aggregated[aggregated["WindSpeed_count"] >= threshold].copy()
    rejected = len(aggregated) - len(accepted)

    if "StatusAnlage" in accepted.columns:
        accepted["Status_rounded"] = accepted["StatusAnlage"].round()

    # Janelas sem nenhuma leitura viram NaN no resample; já foram removidas pelo
    # filtro de qualidade, mas uma coluna esparsa pode sobreviver.
    accepted = accepted.dropna()

    validate_scada_frame(accepted, contract="scada.processed.v1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(output_path)

    normal = int((accepted["Status_rounded"] == PRODUCING_STATUS).sum())
    return IngestionResult(
        raw_rows=len(raw),
        resampled_rows=len(accepted),
        rejected_by_quality=rejected,
        normal_operation_rows=normal,
        output_path=output_path,
    )
