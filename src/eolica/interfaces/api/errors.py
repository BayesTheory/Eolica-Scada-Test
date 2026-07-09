"""Tradução de erros de domínio para HTTP, em um único lugar.

Formato de resposta: RFC 9457 (Problem Details for HTTP APIs). O v1 devolvia
`{"erro": "..."}` em alguns caminhos, `HTTPException(detail=...)` em outros e
500 com stack trace no resto — três formatos para o mesmo cliente.

Este módulo é a **única** fronteira onde uma exceção vira status code. Nenhum
router escreve `HTTPException` à mão, e nenhuma camada abaixo conhece HTTP.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from eolica.shared.errors import (
    ContractViolationError,
    DomainError,
    EolicaError,
    InfrastructureError,
    InsufficientDataError,
    InvalidValueError,
    ModelUnavailableError,
    NotFoundError,
)

logger = logging.getLogger(__name__)

PROBLEM_CONTENT_TYPE = "application/problem+json"
ERROR_DOC_BASE = "https://github.com/BayesTheory/Eolica-Scada-Test/blob/main/docs/errors.md"


def _problem(
    *,
    status: HTTPStatus,
    code: str,
    title: str,
    detail: str,
    instance: str,
    extra: dict[str, object] | None = None,
) -> JSONResponse:
    body: dict[str, object] = {
        "type": f"{ERROR_DOC_BASE}#{code}",
        "title": title,
        "status": int(status),
        "detail": detail,
        "instance": instance,
        "code": code,
    }
    if extra:
        body.update(extra)
    return JSONResponse(status_code=int(status), content=body, media_type=PROBLEM_CONTENT_TYPE)


def register_exception_handlers(app: FastAPI) -> None:
    """Liga cada família de erro ao seu status HTTP."""

    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        """404 — o recurso não existe.

        O caso que o v1 devolvia como 500, e que o prompt do co-piloto
        descrevia ao operador como "dados corrompidos".
        """
        return _problem(
            status=HTTPStatus.NOT_FOUND,
            code="not-found",
            title="Recurso não encontrado",
            detail=exc.message,
            instance=request.url.path,
            extra={"resource": exc.resource, "identifier": str(exc.identifier)},
        )

    @app.exception_handler(InsufficientDataError)
    async def _insufficient(request: Request, exc: InsufficientDataError) -> JSONResponse:
        """422 — o recurso existe, mas não dá para processá-lo como pedido."""
        return _problem(
            status=HTTPStatus.UNPROCESSABLE_ENTITY,
            code="insufficient-data",
            title="Dados insuficientes para a análise",
            detail=exc.message,
            instance=request.url.path,
            extra={"required": exc.required, "available": exc.available},
        )

    @app.exception_handler(InvalidValueError)
    async def _invalid_value(request: Request, exc: InvalidValueError) -> JSONResponse:
        return _problem(
            status=HTTPStatus.BAD_REQUEST,
            code="invalid-value",
            title="Valor inválido",
            detail=exc.message,
            instance=request.url.path,
        )

    @app.exception_handler(DomainError)
    async def _domain(request: Request, exc: DomainError) -> JSONResponse:
        return _problem(
            status=HTTPStatus.BAD_REQUEST,
            code="domain-rule-violated",
            title="Regra de domínio violada",
            detail=exc.message,
            instance=request.url.path,
        )

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(request: Request, exc: ModelUnavailableError) -> JSONResponse:
        """503 — e não 500: é transitório e o cliente deve tentar de novo."""
        logger.error("modelo indisponível", extra={"context": exc.context})
        return _problem(
            status=HTTPStatus.SERVICE_UNAVAILABLE,
            code="model-unavailable",
            title="Modelo indisponível",
            detail=exc.message,
            instance=request.url.path,
        )

    @app.exception_handler(ContractViolationError)
    async def _contract(request: Request, exc: ContractViolationError) -> JSONResponse:
        """500 — contrato violado é sempre bug nosso, e precisa gritar."""
        logger.error(
            "violação de contrato de dado",
            extra={"contract": exc.contract, "violations": exc.violations},
        )
        return _problem(
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="data-contract-violated",
            title="Contrato de dado violado",
            detail="Os dados de entrada não satisfazem o contrato declarado.",
            instance=request.url.path,
        )

    @app.exception_handler(InfrastructureError)
    async def _infrastructure(request: Request, exc: InfrastructureError) -> JSONResponse:
        logger.error("falha de infraestrutura", extra={"context": exc.context})
        return _problem(
            status=HTTPStatus.SERVICE_UNAVAILABLE,
            code="infrastructure-unavailable",
            title="Dependência externa indisponível",
            detail=exc.message,
            instance=request.url.path,
        )

    @app.exception_handler(EolicaError)
    async def _fallback(request: Request, exc: EolicaError) -> JSONResponse:
        """Rede de segurança. `detail` é genérico de propósito.

        Mensagem de exceção pode conter caminho de arquivo, host interno ou
        trecho de query — nada disso atravessa a fronteira. O detalhe fica no
        log, correlacionável pelo path e pelo timestamp.
        """
        logger.exception("erro não tratado", extra={"context": exc.context})
        return _problem(
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="internal-error",
            title="Erro interno",
            detail="A requisição não pôde ser processada.",
            instance=request.url.path,
        )


ExceptionHandler = Callable[[Request, Exception], Awaitable[JSONResponse]]
