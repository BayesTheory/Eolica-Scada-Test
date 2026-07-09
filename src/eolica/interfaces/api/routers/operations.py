"""Endpoints operacionais: liveness, readiness, drift e métricas.

A distinção entre liveness e readiness não é cerimônia de Kubernetes: são
perguntas diferentes. "O processo está vivo?" decide se o orquestrador
reinicia o container. "Ele consegue atender?" decide se recebe tráfego. O v1
não tinha nenhuma das duas — a forma de descobrir que o modelo não carregou era
uma requisição de usuário falhando.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from eolica import __version__
from eolica.interfaces.api.container import Container
from eolica.interfaces.api.dependencies import get_container
from eolica.interfaces.api.schemas import DriftResponse, LivenessResponse, ReadinessResponse

router = APIRouter(tags=["operação"])


@router.get("/health/live", response_model=LivenessResponse, summary="Liveness probe")
def liveness() -> LivenessResponse:
    """O processo está de pé.

    Deliberadamente não toca em modelo, disco nem rede: uma dependência lenta
    não pode fazer o orquestrador matar um processo que está perfeitamente vivo.
    """
    return LivenessResponse(version=__version__)


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    responses={503: {"description": "A aplicação não está pronta para receber tráfego"}},
)
def readiness(
    container: Annotated[Container, Depends(get_container)], response: Response
) -> ReadinessResponse:
    """A aplicação consegue atender: dados carregados e detector calibrado.

    Enquanto o limiar não foi calculado, o serviço não recebe tráfego. No v1 o
    limiar era calculado na primeira requisição do usuário — que pagava a
    varredura de 32 mil registros como latência.
    """
    ready = container.is_ready
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    failed = [name for name, ok in container.checks.items() if not ok]
    return ReadinessResponse(
        ready=ready,
        checks=dict(container.checks),
        detail=None if ready else f"checagens falhando: {', '.join(failed)}",
    )


@router.get(
    "/api/v1/drift",
    response_model=DriftResponse,
    tags=["monitoramento"],
    summary="Drift entre a distribuição de referência e a recente",
    responses={422: {"description": "Histórico insuficiente para comparar dois períodos"}},
)
def drift(container: Annotated[Container, Depends(get_container)]) -> DriftResponse:
    """PSI por feature, com veredito agregado.

    `requires_action` só fica verdadeiro em drift severo (PSI > 0.25). Drift
    moderado pede investigação — sazonalidade de vento move distribuição todo
    trimestre sem que o modelo tenha piorado.
    """
    return DriftResponse.from_domain(container.drift_use_case().execute())
