"""Fixtures compartilhadas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "data" / "samples" / "scada_sample.csv"


@pytest.fixture(scope="session")
def sample_csv_path() -> Path:
    """Recorte real de telemetria versionado no repositório.

    Duas semanas de dados verdadeiros (1395 leituras) escolhidas por conterem
    os casos que quebram implementações ingênuas: gaps temporais, potência
    negativa e códigos de status indocumentados. Ver `scripts/make_sample.py`.
    """
    if not SAMPLE_CSV.exists():
        pytest.fail(f"Sample ausente em {SAMPLE_CSV}. Gere com: python scripts/make_sample.py")
    return SAMPLE_CSV


@pytest.fixture(scope="session")
def sample_frame(sample_csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(sample_csv_path, index_col="Datetime", parse_dates=True)
    frame.index = frame.index.tz_localize("UTC")
    return frame.sort_index()
