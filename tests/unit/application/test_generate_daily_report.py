"""Caso de uso `GenerateDailyReport`, testado inteiramente com fakes.

Nenhum destes testes toca disco, rede ou MLflow — e cobrem os caminhos de erro
que no v1 viravam HTTP 500 indiscriminadamente.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from tests.fakes import (
    SAMPLING_INTERVAL,
    BrokenForecastModel,
    FakeForecastModel,
    FakeReconstructionModel,
    InMemoryScadaRepository,
    make_day,
)

from eolica.application.use_cases import GenerateDailyReport
from eolica.domain.forecasting import Horizon
from eolica.domain.health import AnomalyThreshold, HealthStatus, ThresholdMethod
from eolica.shared.errors import InsufficientDataError, NotFoundError

DAY = date(2022, 1, 20)
PREVIOUS_DAY = date(2022, 1, 19)
THRESHOLD = AnomalyThreshold(value=1.0, method=ThresholdMethod.PERCENTILE, parameter=99.5)


def build(
    *,
    repository: InMemoryScadaRepository,
    health_model: FakeReconstructionModel | None = None,
    forecast_model: FakeForecastModel | None = None,
    persistence_window: int = 3,
) -> GenerateDailyReport:
    return GenerateDailyReport(
        readings=repository,
        health_model=health_model or FakeReconstructionModel(window_size=6),
        forecast_model=forecast_model or FakeForecastModel(),
        threshold=THRESHOLD,
        persistence_window=persistence_window,
        sampling_interval=SAMPLING_INTERVAL,
        forecast_horizon=Horizon(steps=1, step=SAMPLING_INTERVAL),
    )


class TestDiaAusente:
    def test_dia_fora_do_acervo_e_not_found_e_nao_erro_de_servidor(self) -> None:
        """O bug mais visível do v1.

        `df_principal.loc["2022-02-08"]` levantava `KeyError`, que subia como
        500. O prompt do co-piloto chegou a documentar isso como característica
        do sistema: "Se a API retornar um erro interno (status 500) para uma
        data específica (como 2022-02-08), informe ao usuário que os dados para
        esse dia específico estão corrompidos".

        Os dados não estavam corrompidos. O dia simplesmente não existia.
        """
        repository = InMemoryScadaRepository(make_day(DAY))
        with pytest.raises(NotFoundError) as exc:
            build(repository=repository).execute(date(2030, 1, 1))
        assert exc.value.resource == "Telemetria"

    def test_dia_ausente_carrega_o_identificador_pedido(self) -> None:
        repository = InMemoryScadaRepository(make_day(DAY))
        with pytest.raises(NotFoundError, match="2030-01-01"):
            build(repository=repository).execute(date(2030, 1, 1))


class TestDiaSaudavel:
    def test_dia_limpo_resulta_em_ok(self) -> None:
        repository = InMemoryScadaRepository(make_day(DAY))
        report = build(repository=repository).execute(DAY)
        assert report.health.status is HealthStatus.OK

    def test_relatorio_traz_o_periodo_coberto_pelo_acervo(self) -> None:
        repository = InMemoryScadaRepository(make_day(DAY))
        report = build(repository=repository).execute(DAY)
        start, end = report.data_range
        assert start.date() == DAY
        assert end.date() == DAY

    def test_cobertura_de_dia_completo_e_total(self) -> None:
        repository = InMemoryScadaRepository(make_day(DAY, count=144))
        report = build(repository=repository).execute(DAY)
        assert report.coverage.completeness == pytest.approx(1.0)
        assert report.coverage.is_fragmented is False


class TestFragmentacao:
    def _fragmented_day(self) -> InMemoryScadaRepository:
        """Dois blocos de 20 leituras separados por um buraco de 5 horas."""
        first = make_day(DAY, count=20, start_step=0)
        second = make_day(DAY, count=20, start_step=50)
        return InMemoryScadaRepository([*first, *second])

    def test_dia_fragmentado_e_analisado_por_segmento(self) -> None:
        """O v1 passaria uma janela atravessando o buraco para o autoencoder."""
        model = FakeReconstructionModel(window_size=6)
        report = build(repository=self._fragmented_day(), health_model=model).execute(DAY)
        assert report.coverage.analysed_segments == 2
        assert report.coverage.is_fragmented is True

    def test_cada_segmento_chega_intacto_ao_modelo(self) -> None:
        model = FakeReconstructionModel(window_size=6)
        build(repository=self._fragmented_day(), health_model=model).execute(DAY)
        assert [len(w) for w in model.calls] == [20, 20]
        for window in model.calls:
            assert window.end - window.start == 19 * SAMPLING_INTERVAL

    def test_cobertura_parcial_e_reportada(self) -> None:
        report = build(repository=self._fragmented_day()).execute(DAY)
        assert report.coverage.readings == 40
        assert report.coverage.completeness == pytest.approx(40 / 144)

    def test_segmentos_curtos_demais_sao_descartados_e_contabilizados(self) -> None:
        """Um trecho de 3 leituras não cabe numa janela de 6: some da análise,
        mas aparece em `discarded_readings` para não sumir do relatório."""
        long_block = make_day(DAY, count=20, start_step=0)
        crumb = make_day(DAY, count=3, start_step=60)
        repository = InMemoryScadaRepository([*long_block, *crumb])
        report = build(repository=repository).execute(DAY)
        assert report.coverage.analysed_segments == 1
        assert report.coverage.discarded_readings == 3

    def test_dia_inteiro_fragmentado_demais_e_insufficient_data(self) -> None:
        """Existe dado, mas nenhum trecho contíguo serve. Não é 404 nem 500."""
        crumbs = [*make_day(DAY, count=2, start_step=0), *make_day(DAY, count=2, start_step=40)]
        repository = InMemoryScadaRepository(crumbs)
        with pytest.raises(InsufficientDataError):
            build(repository=repository).execute(DAY)


class TestRegraDeManutencaoNoCasoDeUso:
    def _repo_with_two_days(self) -> InMemoryScadaRepository:
        return InMemoryScadaRepository([*make_day(PREVIOUS_DAY), *make_day(DAY)])

    def test_anomalia_nos_dois_dias_resulta_em_manutencao(self) -> None:
        model = FakeReconstructionModel(window_size=6, error_for=lambda _: 5.0)
        report = build(repository=self._repo_with_two_days(), health_model=model).execute(DAY)
        assert report.health.status is HealthStatus.UNDER_MAINTENANCE

    def test_anomalia_so_hoje_resulta_em_alerta(self) -> None:
        repository = InMemoryScadaRepository(make_day(DAY))
        model = FakeReconstructionModel(window_size=6, error_for=lambda _: 5.0)
        report = build(repository=repository, health_model=model).execute(DAY)
        assert report.health.status is HealthStatus.ALERT

    def test_vespera_ausente_marca_periodo_anterior_como_desconhecido(self) -> None:
        """Sem véspera no acervo, não se conclui manutenção — e isso fica
        explícito no veredito, em vez de virar o sentinela -1 do v1."""
        repository = InMemoryScadaRepository(make_day(DAY))
        model = FakeReconstructionModel(window_size=6, error_for=lambda _: 5.0)
        report = build(repository=repository, health_model=model).execute(DAY)
        assert report.health.previous_period_known is False


class TestDegradacaoDaPrevisao:
    def test_falha_de_previsao_nao_derruba_o_relatorio(self) -> None:
        """Saúde é a parte crítica; previsão é acessória e pode faltar."""
        repository = InMemoryScadaRepository(make_day(DAY))
        report = build(repository=repository, forecast_model=BrokenForecastModel()).execute(DAY)
        assert report.forecast is None
        assert report.health.status is HealthStatus.OK

    def test_falha_de_previsao_reporta_o_motivo(self) -> None:
        """O v1 devolvia a string "Indisponível" no campo numérico `previsao_kw`,
        quebrando o tipo e sem dizer o porquê."""
        repository = InMemoryScadaRepository(make_day(DAY))
        report = build(repository=repository, forecast_model=BrokenForecastModel()).execute(DAY)
        assert report.forecast_unavailable_reason is not None
        assert "observações" in report.forecast_unavailable_reason

    def test_previsao_bem_sucedida_traz_versao_do_modelo(self) -> None:
        repository = InMemoryScadaRepository(make_day(DAY))
        report = build(repository=repository).execute(DAY)
        assert report.forecast is not None
        assert report.forecast.model_version == "fake-forecaster@1"

    def test_previsao_aponta_para_o_instante_seguinte(self) -> None:
        repository = InMemoryScadaRepository(make_day(DAY))
        report = build(repository=repository).execute(DAY)
        assert report.forecast is not None
        expected = datetime(2022, 1, 20, 23, 50, tzinfo=UTC) + timedelta(minutes=10)
        assert report.forecast.target_time == expected
