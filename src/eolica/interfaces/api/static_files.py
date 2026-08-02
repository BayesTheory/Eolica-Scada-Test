"""Serviço do build estático do frontend, pela própria API.

Um único deployable em vez de dois. A alternativa — front num bucket ou CDN
separado — traria CORS, um segundo pipeline de deploy e a possibilidade de as
duas pontas ficarem em versões diferentes. Para um serviço que responde a um
punhado de rotas, o custo não se paga.

Como front e API compartilham origem, `EOLICA_CORS_ALLOWED_ORIGINS` fica vazio
em produção. CORS habilitado só em desenvolvimento é a forma clássica de
descobrir o problema no deploy.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from eolica.infrastructure.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_BUILD_DIR = PROJECT_ROOT / "frontend" / "dist"
"""Onde procurar o build.

Pode ser sobrescrito por `EOLICA_FRONTEND_DIR`. Num container o pacote está em
site-packages e a descoberta de raiz cai no `WORKDIR`, então o default acerta —
mas apontar explicitamente é mais barato de depurar que confiar em heurística.
"""

# Prefixos reservados à API. Uma rota do frontend nunca pode sombreá-los.
API_PREFIXES = ("/api", "/health", "/metrics", "/docs", "/redoc", "/openapi.json")


def mount_frontend(app: FastAPI, build_dir: Path = DEFAULT_BUILD_DIR) -> bool:
    """Serve o build do frontend, se ele existir.

    Ausência do build não é erro: a API é útil sozinha, e o container de treino
    não precisa de interface. Mas a ausência é registrada em log — descobrir que
    o front não subiu por uma tela em branco é pior que ler uma linha no boot.

    Returns:
        True se o build foi montado.
    """
    index = build_dir / "index.html"
    if not index.is_file():
        logger.info(
            "build do frontend ausente; servindo apenas a API",
            extra={"expected": str(build_dir)},
        )
        return False

    # Os assets do Vite têm hash no nome, então são imutáveis e podem ser
    # cacheados agressivamente pelo navegador.
    app.mount(
        "/assets",
        StaticFiles(directory=build_dir / "assets"),
        name="frontend-assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        """Devolve o index para qualquer rota não reservada.

        É o comportamento esperado de uma SPA: o roteamento acontece no cliente,
        então uma URL profunda recarregada precisa receber o index em vez de 404.

        A guarda de prefixo importa: sem ela, esta rota engoliria um typo em
        `/api/v1/reprts/...` e devolveria HTML com status 200 — o cliente
        receberia uma página onde esperava JSON, e o erro viraria um parse
        failure obscuro em vez de um 404 honesto.
        """
        if full_path.startswith(tuple(prefix.lstrip("/") for prefix in API_PREFIXES)):
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Rota de API inexistente")
        return FileResponse(index)

    logger.info("frontend montado", extra={"build_dir": str(build_dir)})
    return True
