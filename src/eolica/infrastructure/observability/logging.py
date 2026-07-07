"""Logging estruturado.

O v1 usava `print()` com emoji — inclusive para erro crítico
(`print("❌ ERRO FATAL ao inicializar os especialistas")`). Em container isso
vai para stdout como texto solto: não é indexável, não tem nível, não tem
timestamp e não correlaciona com uma requisição.

Aqui: JSON em produção (parseável por qualquer coletor), texto colorido em
desenvolvimento.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LogFormat = Literal["json", "console"]

_configured = False


def configure_logging(*, level: LogLevel = "INFO", fmt: LogFormat = "json") -> None:
    """Configura structlog e a stdlib para saírem no mesmo formato.

    Idempotente: `create_app()` pode ser chamada várias vezes num mesmo processo
    de teste sem empilhar handlers.
    """
    global _configured
    if _configured:
        return

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level),
        force=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Logger estruturado nomeado."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
