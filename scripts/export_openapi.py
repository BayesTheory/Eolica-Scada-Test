"""Exporta o schema OpenAPI para arquivo, sem subir servidor.

É o artefato de fronteira entre backend e frontend. Gerá-lo offline importa por
dois motivos:

1. O CI consegue verificar que os tipos TypeScript commitados batem com o schema
   atual — sem orquestrar um servidor de verdade só para isso.
2. O `openapi.json` versionado torna visível, no diff do PR, quando um contrato
   público muda. Renomear um campo deixa de ser detalhe de implementação e passa
   a ser uma linha vermelha que alguém revisa.

Uso:
    python scripts/export_openapi.py [destino.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eolica.infrastructure.config import Settings
from eolica.interfaces.api.app import create_app

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DESTINATION = ROOT / "frontend" / "openapi.json"


def main() -> int:
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DESTINATION

    # `create_app` não dispara o lifespan: montar a aplicação não toca disco nem
    # rede, então o schema sai sem carregar dado nenhum. Era impossível no v1,
    # onde importar o módulo da API já abria o CSV e conectava no MLflow.
    app = create_app(Settings(environment="ci"))
    schema = app.openapi()

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    paths = len(schema.get("paths", {}))
    schemas = len(schema.get("components", {}).get("schemas", {}))
    print(f"openapi exportado para {destination.relative_to(ROOT)}")
    print(f"  rotas ....... {paths}")
    print(f"  schemas ..... {schemas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
