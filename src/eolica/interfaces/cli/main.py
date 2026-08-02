"""CLI do projeto.

Mesmos casos de uso da API, outra porta de entrada. Nenhuma regra de negócio
mora aqui — se morasse, CLI e HTTP divergiriam, que é exatamente o que
aconteceu no v1 com a regra de "em manutenção" existindo só no cliente de chat.

Substitui o `main.py` do v1, cujo comando `process_data` chamava uma função
inexistente.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import typer

from eolica import __version__
from eolica.infrastructure.config import PROJECT_ROOT, Settings
from eolica.infrastructure.observability import configure_logging
from eolica.infrastructure.persistence.ingestion import ingest_scada
from eolica.interfaces.api.container import build_container
from eolica.shared.errors import EolicaError

app = typer.Typer(
    name="eolica",
    help="Monitoramento preditivo de turbinas eólicas a partir de telemetria SCADA.",
    no_args_is_help=True,
    add_completion=False,
)


def _settings() -> Settings:
    settings = Settings()
    configure_logging(level=settings.log_level, fmt="console")
    return settings


def _fail(exc: EolicaError) -> None:
    typer.secho(f"erro: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Mostra a versão instalada."""
    typer.echo(f"eolica-scada {__version__}")


@app.command()
def ingest(
    raw: Annotated[
        Path, typer.Option(help="CSV SCADA bruto a 1 Hz, baixado do Zenodo")
    ] = PROJECT_ROOT / "data" / "raw" / "Aventa_AV7_IET_OST_SCADA.csv",
    output: Annotated[Path, typer.Option(help="Destino do CSV reamostrado")] = PROJECT_ROOT
    / "data"
    / "processed"
    / "scada_resampled_10min_base.csv",
) -> None:
    """Reamostra o SCADA bruto para a grade de 10 minutos e valida o contrato."""
    try:
        result = ingest_scada(raw_path=raw, output_path=output)
    except EolicaError as exc:
        _fail(exc)
        return

    typer.secho("ingestão concluída", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  linhas brutas ........... {result.raw_rows:,}")
    typer.echo(f"  linhas reamostradas ..... {result.resampled_rows:,}")
    typer.echo(
        f"  rejeitadas por qualidade  {result.rejected_by_quality:,} "
        f"({result.quality_rejection_ratio:.1%})"
    )
    typer.echo(f"  em operação normal ...... {result.normal_operation_rows:,}")
    typer.echo(f"  destino ................. {result.output_path}")


@app.command()
def calibrate() -> None:
    """Calibra o detector e mostra o limiar resultante, sem subir a API."""
    try:
        container = build_container(_settings())
    except EolicaError as exc:
        _fail(exc)
        return

    calibration = container.calibration
    typer.secho("calibração concluída", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  limiar .................. {calibration.threshold.value:.6f}")
    typer.echo(
        f"  método .................. {calibration.threshold.method} "
        f"(p={calibration.threshold.parameter})"
    )
    typer.echo(f"  janelas de referência ... {calibration.reference_windows:,}")
    typer.echo(f"  erros de referência ..... {calibration.reference_errors:,}")
    typer.echo(f"  duração ................. {calibration.duration_seconds:.2f}s")


@app.command()
def report(
    # Typer não converte `date` (só `datetime`), então o parsing é feito aqui e
    # o caso de uso continua recebendo o tipo certo.
    day: Annotated[
        datetime, typer.Argument(help="Dia a analisar (YYYY-MM-DD)", formats=["%Y-%m-%d"])
    ],
) -> None:
    """Gera o relatório diário de saúde e previsão no terminal."""
    target: date = day.date()
    try:
        container = build_container(_settings())
        result = container.daily_report_use_case().execute(target)
    except EolicaError as exc:
        _fail(exc)
        return

    colour = {
        "OK": typer.colors.GREEN,
        "ALERTA": typer.colors.RED,
        "EM_MANUTENCAO": typer.colors.YELLOW,
    }.get(str(result.health.status), typer.colors.WHITE)

    typer.secho(f"\n{target}  —  {result.health.status}", fg=colour, bold=True)
    typer.echo(f"  {result.health.reason}\n")
    typer.echo(f"  janelas avaliadas ....... {result.health.evaluated_windows}")
    typer.echo(f"  acima do limiar ......... {result.health.exceedances}")
    typer.echo(f"  anomalias sustentadas ... {result.health.sustained_anomalies}")
    typer.echo(f"  limiar .................. {result.health.threshold.value:.6f}")
    typer.echo(
        "  véspera ................. "
        + (
            "desconhecida"
            if not result.health.previous_period_known
            else f"{result.health.previous_period_anomalies} anomalia(s)"
        )
    )
    typer.echo(
        f"\n  cobertura ............... {result.coverage.completeness:.1%} "
        f"({result.coverage.readings}/{result.coverage.expected_readings} leituras, "
        f"{result.coverage.analysed_segments} segmento(s))"
    )
    if result.forecast is not None:
        typer.echo(
            f"  previsão ................ {result.forecast.for_display():.3f} kW "
            f"@ {result.forecast.target_time:%Y-%m-%d %H:%M} "
            f"[{result.forecast.model_version}]"
        )
    else:
        typer.echo(
            f"  previsão ................ indisponível: {result.forecast_unavailable_reason}"
        )


@app.command()
def drift() -> None:
    """Compara a distribuição de referência com a recente (PSI por feature)."""
    try:
        container = build_container(_settings())
        result = container.drift_use_case().execute()
    except EolicaError as exc:
        _fail(exc)
        return

    colour = {"none": typer.colors.GREEN, "moderate": typer.colors.YELLOW}.get(
        str(result.severity), typer.colors.RED
    )
    typer.secho(f"\ndrift: {result.severity}", fg=colour, bold=True)
    typer.echo(f"  requer ação: {'sim' if result.requires_action else 'não'}\n")
    for name, score in sorted(result.scores.items(), key=lambda item: -item[1].value):
        typer.echo(f"  {name:<24} PSI={score.value:8.4f}  {score.severity}")


@app.command()
def backtest() -> None:
    """Mede quanto a janela de persistência vale sobre o histórico inteiro.

    Compara valores de janela quanto a episódios de alarme, alarmes falsos e
    detecções perdidas. O v1 declarava `persistence_window: 6` no config sem
    nunca lê-lo — e sem nenhum número que justificasse o 6.
    """
    try:
        container = build_container(_settings())
        summary = container.backtest_use_case().execute()
    except EolicaError as exc:
        _fail(exc)
        return

    typer.secho("\nbacktest da janela de persistência", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        f"  {summary.total_readings:,} leituras | "
        f"{summary.analysed_segments} segmentos contíguos | "
        f"{summary.report.evaluated_windows:,} janelas avaliadas"
    )
    typer.echo(f"  limiar: {summary.report.threshold.value:.6f}")
    typer.echo(f"  janelas com falha reportada pelo SCADA: {summary.real_event_windows:,}\n")

    header = f"  {'janela':>7} {'episódios':>10} {'precisão':>9} {'recall':>7} {'alarme falso':>13}"
    typer.echo(header)
    typer.echo("  " + "-" * (len(header) - 2))
    for outcome in summary.report.outcomes:
        typer.echo(
            f"  {outcome.persistence_window:>7} {outcome.episodes:>10,} "
            f"{outcome.metrics.precision:>8.1%} {outcome.recall:>6.1%} "
            f"{outcome.false_alarm_rate:>12.2%}"
        )

    baseline = summary.report.outcomes[0].persistence_window
    for outcome in summary.report.outcomes[1:]:
        avoided = summary.report.false_alarms_avoided(
            baseline=baseline, candidate=outcome.persistence_window
        )
        lost = summary.report.detections_lost(
            baseline=baseline, candidate=outcome.persistence_window
        )
        typer.echo(
            f"\n  janela {outcome.persistence_window} vs {baseline}: "
            f"evita {avoided:,} janelas de alarme falso, perde {lost:,} de detecção"
        )


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Interface de bind")] = None,
    port: Annotated[int | None, typer.Option(help="Porta TCP")] = None,
    reload: Annotated[bool, typer.Option(help="Recarrega ao salvar (só em dev)")] = False,
) -> None:
    """Sobe a API HTTP."""
    import uvicorn

    settings = _settings()
    uvicorn.run(
        "eolica.interfaces.api.app:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
        log_config=None,
    )


if __name__ == "__main__":
    app()
