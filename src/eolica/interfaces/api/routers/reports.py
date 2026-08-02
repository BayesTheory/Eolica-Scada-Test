"""Relatório diário de saúde e previsão."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from eolica.interfaces.api.container import Container
from eolica.interfaces.api.dependencies import get_container
from eolica.interfaces.api.schemas import CoverageTimelineResponse, DailyReportResponse

router = APIRouter(prefix="/api/v1", tags=["relatórios"])


@router.get(
    "/reports/{report_date}",
    response_model=DailyReportResponse,
    summary="Relatório diário de saúde e previsão",
    responses={
        404: {"description": "A data não existe no acervo de telemetria"},
        422: {"description": "A data existe mas não há dado contíguo suficiente para analisar"},
        503: {"description": "Modelo ou fonte de dados indisponível"},
    },
)
def daily_report(
    report_date: Annotated[date, Path(description="Dia a analisar, em ISO-8601 (YYYY-MM-DD)")],
    container: Annotated[Container, Depends(get_container)],
) -> DailyReportResponse:
    """Analisa um dia de telemetria e devolve saúde, cobertura e previsão.

    A data vai no path e é tipada como `date`: o FastAPI valida o formato antes
    de a requisição chegar aqui. No v1 a data era uma query string livre
    (`?data_string=...`) usada direto como chave de `.loc` num DataFrame — o que
    fazia `2022-02-31` e `; DROP` seguirem o mesmo caminho de código.

    Todo caminho de erro é tipado: dia inexistente é 404, dia fragmentado
    demais é 422, registry fora do ar é 503. Nenhum deles é 500.
    """
    report = container.daily_report_use_case().execute(report_date)
    return DailyReportResponse.from_domain(report)


@router.get(
    "/coverage",
    response_model=CoverageTimelineResponse,
    summary="Cobertura de telemetria dia a dia",
    responses={400: {"description": "Período inválido ou maior que o máximo permitido"}},
)
def coverage_timeline(
    container: Annotated[Container, Depends(get_container)],
    start: Annotated[
        date | None, Query(description="Início do período (padrão: início do acervo)")
    ] = None,
    end: Annotated[date | None, Query(description="Fim do período (padrão: fim do acervo)")] = None,
) -> CoverageTimelineResponse:
    """Fração de cada dia com medição, e em quantos trechos contíguos ela vem.

    Não roda modelo nenhum — é varredura de índice, e responde em milissegundos
    mesmo cobrindo o acervo inteiro.

    É a visão que torna a fragmentação da série visível: ver que 2022-01-20 tem
    43,8% de cobertura é útil, ver que dezenas de dias têm o mesmo problema é o
    que muda a leitura de todos os vereditos do sistema.
    """
    summary = container.coverage_use_case().execute(start=start, end=end)
    return CoverageTimelineResponse.from_domain(summary)
