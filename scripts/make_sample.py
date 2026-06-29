"""Gera o recorte versionado de dados SCADA usado por testes e pelo modo demo.

O dataset completo (~65k linhas, 25 MB) não é versionado: vem do Zenodo e é
reproduzido pelo pipeline de ingestão. Mas testes precisam de dado real e
determinístico, então commitamos uma janela pequena e representativa.

A janela é escolhida para conter:
  - operação normal (status 10),
  - parada/idle (status 9) e falha (status 13),
  - pelo menos um gap temporal (para exercitar o windowing gap-aware),
  - potência negativa (consumo parasita), que é o caso que o v1 mascarava.

Uso:
    python scripts/make_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "processed" / "scada_resampled_10min_base.csv"
DESTINATION = ROOT / "data" / "samples" / "scada_sample.csv"

# Janela com maior diversidade de status no dataset (ver notebooks/01).
WINDOW_START = "2022-01-14"
WINDOW_END = "2022-01-27"


def main() -> int:
    if not SOURCE.exists():
        print(f"ERRO: fonte não encontrada em {SOURCE}", file=sys.stderr)
        print("Rode o pipeline de ingestão antes: eolica ingest", file=sys.stderr)
        return 1

    frame = pd.read_csv(SOURCE, index_col="Datetime", parse_dates=True)
    sample = frame.loc[WINDOW_START:WINDOW_END].copy()

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(DESTINATION, float_format="%.6f")

    gaps = sample.index.to_series().diff().dropna()
    irregular = int((gaps != pd.Timedelta(minutes=10)).sum())

    print(f"sample salvo em {DESTINATION.relative_to(ROOT)}")
    print(f"  linhas .............. {len(sample)}")
    print(f"  período ............. {sample.index.min()} -> {sample.index.max()}")
    print(f"  tamanho ............. {DESTINATION.stat().st_size / 1024:.0f} KB")
    print(f"  status presentes .... {sorted(sample['Status_rounded'].unique())}")
    print(f"  gaps (!= 10min) ..... {irregular}")
    print(f"  potência negativa ... {int((sample['PowerOutput'] < 0).sum())} linhas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
