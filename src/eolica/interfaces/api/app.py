"""Fábrica da aplicação FastAPI.

`create_app()` é uma função, não um `app` global montado no import do módulo.
A diferença é o que torna o serviço testável: um teste pode construir a
aplicação com um container próprio, sem CSV e sem MLflow, e sem que importar o
módulo dispare I/O.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from eolica import __version__
from eolica.infrastructure.config import Settings
from eolica.infrastructure.observability import configure_logging, mount_metrics
from eolica.interfaces.api.container import Container, build_container
from eolica.interfaces.api.errors import register_exception_handlers
from eolica.interfaces.api.routers import operations, reports

logger = logging.getLogger(__name__)

DESCRIPTION = """
Monitoramento preditivo de turbinas eólicas a partir de telemetria SCADA.

**Detecção de anomalia** por erro de reconstrução, com janela de persistência
para não alarmar em ruído de sensor. **Previsão de geração** para o passo
seguinte. **Drift** entre a distribuição de referência e a recente.

Erros seguem RFC 9457 (`application/problem+json`).
"""


def create_app(settings: Settings | None = None, *, container: Container | None = None) -> FastAPI:
    """Monta a aplicação.

    Args:
        settings: configuração; se omitida, é lida do ambiente.
        container: dependências já construídas. Usado por testes para injetar
            fakes — em produção fica `None` e o `lifespan` monta o container
            real, calibrando o detector antes de o serviço aceitar tráfego.
    """
    resolved = settings or Settings()
    configure_logging(level=resolved.log_level, fmt=resolved.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Carrega dados e calibra o detector antes do primeiro request.

        Uma falha aqui derruba a subida — que é o comportamento certo. O v1
        chamava `sys.exit(1)` no meio do import do módulo, o que matava
        qualquer processo que apenas importasse o arquivo, inclusive o pytest.
        """
        app.state.container = container or build_container(resolved)
        logger.info(
            "aplicação pronta",
            extra={
                "version": __version__,
                "environment": resolved.environment,
                "readings": len(app.state.container.repository),
            },
        )
        yield
        app.state.container = None

    app = FastAPI(
        title="Eólica SCADA",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    if resolved.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.cors_allowed_origins),
            allow_methods=["GET"],
            allow_headers=["*"],
        )

    register_exception_handlers(app)
    app.include_router(operations.router)
    app.include_router(reports.router)

    if resolved.enable_metrics:
        mount_metrics(app)

    return app
