"""Contrato de dado validado contra telemetria **real**.

Testes de unidade com dado sintético provam que o código faz o que o autor
imaginou. Só dado real prova que o autor imaginou o dataset certo — e este
dataset tem três armadilhas que dado sintético bem-comportado nunca teria.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from eolica.domain.turbine import OperatingStatus, ReadingWindow
from eolica.infrastructure.persistence import (
    EXCLUDED_CHANNELS,
    REQUIRED_COLUMNS,
    CsvScadaRepository,
    validate_scada_frame,
)
from eolica.shared.errors import ContractViolationError, InsufficientDataError, NotFoundError

pytestmark = pytest.mark.contract

SAMPLING_INTERVAL = timedelta(minutes=10)


@pytest.fixture(scope="module")
def repository(sample_csv_path: Path) -> CsvScadaRepository:
    """O repositório real sobre o sample versionado.

    Fixture de módulo, e não de classe: duas fixtures homônimas com escopo de
    classe dependendo de uma de sessão disparam um `AssertionError` interno do
    pytest na cadeia de finalizers.
    """
    return CsvScadaRepository.from_path(sample_csv_path)


class TestContratoContraDadoReal:
    def test_o_sample_versionado_satisfaz_o_contrato(self, sample_frame: pd.DataFrame) -> None:
        validate_scada_frame(sample_frame)

    def test_todas_as_colunas_exigidas_existem(self, sample_frame: pd.DataFrame) -> None:
        assert set(REQUIRED_COLUMNS) <= set(sample_frame.columns)

    def test_o_canal_nao_confiavel_nao_entra_no_contrato(self) -> None:
        """`GeneratorSpeed` existe no CSV e é excluído de propósito: o metadado
        do fabricante o marca como `Reliable Measurement = FALSE`."""
        assert EXCLUDED_CHANNELS & set(REQUIRED_COLUMNS) == set()

    def test_coluna_faltando_acusa_violacao_nomeando_a_coluna(
        self, sample_frame: pd.DataFrame
    ) -> None:
        mutilated = sample_frame.drop(columns=["PowerOutput"])
        with pytest.raises(ContractViolationError, match="PowerOutput"):
            validate_scada_frame(mutilated)

    def test_violacoes_sao_acumuladas_e_nao_reportadas_uma_a_uma(
        self, sample_frame: pd.DataFrame
    ) -> None:
        mutilated = sample_frame.drop(columns=["PowerOutput", "WindSpeed", "RotorSpeed"])
        with pytest.raises(ContractViolationError) as exc:
            validate_scada_frame(mutilated)
        assert len(exc.value.violations) == 3

    def test_nulo_injetado_e_detectado(self, sample_frame: pd.DataFrame) -> None:
        corrupted = sample_frame.copy()
        corrupted.loc[corrupted.index[5], "WindSpeed"] = None
        with pytest.raises(ContractViolationError, match="nulo"):
            validate_scada_frame(corrupted)

    def test_indice_desordenado_e_detectado(self, sample_frame: pd.DataFrame) -> None:
        shuffled = sample_frame.iloc[::-1]
        with pytest.raises(ContractViolationError, match="ordenado"):
            validate_scada_frame(shuffled)

    def test_dataframe_vazio_e_detectado(self, sample_frame: pd.DataFrame) -> None:
        with pytest.raises(ContractViolationError, match="nenhuma linha"):
            validate_scada_frame(sample_frame.iloc[0:0])


class TestRepositorioSobreDadoReal:
    def test_carrega_o_sample_inteiro(self, repository: CsvScadaRepository) -> None:
        assert len(repository) == 1395

    def test_intervalo_disponivel_bate_com_o_recorte(self, repository: CsvScadaRepository) -> None:
        start, end = repository.available_range()
        assert start == datetime(2022, 1, 14, 0, 0, tzinfo=UTC)
        assert end == datetime(2022, 1, 27, 23, 50, tzinfo=UTC)

    def test_dia_existente_devolve_leituras(self, repository: CsvScadaRepository) -> None:
        readings = repository.readings_for_day(date(2022, 1, 20))
        assert readings
        assert all(r.timestamp.date() == date(2022, 1, 20) for r in readings)

    def test_dia_fora_do_recorte_e_not_found(self, repository: CsvScadaRepository) -> None:
        """Nunca 500. Este é o caso que o v1 errava."""
        with pytest.raises(NotFoundError):
            repository.readings_for_day(date(2019, 1, 1))

    def test_historico_insuficiente_e_erro_tipado(self, repository: CsvScadaRepository) -> None:
        first_moment = datetime(2022, 1, 14, 0, 10, tzinfo=UTC)
        with pytest.raises(InsufficientDataError):
            repository.readings_before(first_moment, limit=100)

    def test_arquivo_inexistente_e_data_source_error(self, tmp_path: Path) -> None:
        from eolica.shared.errors import DataSourceError

        with pytest.raises(DataSourceError, match="não encontrado"):
            CsvScadaRepository.from_path(tmp_path / "nao-existe.csv")


class TestArmadilhasDoDadoReal:
    """Os três casos que o dado sintético não pega."""

    def test_o_sample_contem_potencia_negativa(self, sample_frame: pd.DataFrame) -> None:
        """24 linhas de consumo parasita — o caso que o prompt do LLM mascarava."""
        assert (sample_frame["PowerOutput"] < 0).sum() > 0

    def test_potencia_negativa_sobrevive_ao_carregamento(
        self, repository: CsvScadaRepository
    ) -> None:
        start, end = repository.available_range()
        readings = repository.readings_between(start, end)
        parasitic = [r for r in readings if r.power.is_parasitic]
        assert parasitic, "o dado bruto não pode ser adulterado no carregamento"
        assert all(r.power.for_display() == 0.0 for r in parasitic)

    def test_o_sample_contem_codigos_de_status_indocumentados(
        self, repository: CsvScadaRepository
    ) -> None:
        """O código 305 aparece no dataset e não está em nenhum metadado."""
        start, end = repository.available_range()
        statuses = {r.status for r in repository.readings_between(start, end)}
        assert OperatingStatus.UNKNOWN in statuses

    def test_o_sample_contem_descontinuidades_temporais(self, sample_frame: pd.DataFrame) -> None:
        deltas = sample_frame.index.to_series().diff().dropna()
        assert (deltas > SAMPLING_INTERVAL).sum() > 0

    def test_janela_ingenua_sobre_o_sample_seria_furada(
        self, repository: CsvScadaRepository
    ) -> None:
        """Prova que o bug do v1 era alcançável com este dado.

        `ReadingWindow.of` recusa a sequência inteira justamente porque ela tem
        buracos — que é o que o `iloc[-n:]` do v1 nunca notou.
        """
        from eolica.shared.errors import InvalidValueError

        start, end = repository.available_range()
        everything = repository.readings_between(start, end)
        with pytest.raises(InvalidValueError, match="descontinuidade"):
            ReadingWindow.of(everything, expected_interval=SAMPLING_INTERVAL)

    def test_segmentacao_recupera_trechos_contiguos_do_dado_real(
        self, repository: CsvScadaRepository
    ) -> None:
        start, end = repository.available_range()
        everything = repository.readings_between(start, end)
        segments = ReadingWindow.split_on_gaps(
            everything, expected_interval=SAMPLING_INTERVAL, min_length=60
        )
        assert len(segments) > 1
        assert sum(len(s) for s in segments) < len(everything), "trechos curtos são descartados"

    def test_operacao_normal_e_um_subconjunto_proprio(self, repository: CsvScadaRepository) -> None:
        normal = repository.normal_operation_readings()
        assert 0 < len(normal) < len(repository)
        assert all(r.status is OperatingStatus.PRODUCING for r in normal)
