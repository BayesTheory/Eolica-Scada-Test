"""Acesso ao container a partir das rotas."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from eolica.shared.errors import InfrastructureError

if TYPE_CHECKING:
    from eolica.interfaces.api.container import Container


def get_container(request: Request) -> Container:
    """Recupera o container montado no `lifespan`.

    Se ele não existe, a aplicação subiu quebrada — e isso é 503, não 500.
    """
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:
        raise InfrastructureError("A aplicação ainda não terminou de inicializar")
    return container
