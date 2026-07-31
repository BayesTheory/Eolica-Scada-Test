"""Backtest da janela de persistência."""

from __future__ import annotations

import pytest

from eolica.domain.health import AnomalyThreshold, ReconstructionError, ThresholdMethod
from eolica.domain.health.backtest import (
    backtest_persistence,
    find_alarm_episodes,
)
from eolica.shared.errors import InsufficientDataError, InvalidValueError

THRESHOLD = AnomalyThreshold(value=1.0, method=ThresholdMethod.PERCENTILE, parameter=99.5)


def _errors(*values: float) -> list[ReconstructionError]:
    return [ReconstructionError(v) for v in values]


class TestEpisodios:
    def test_corrida_longa_o_bastante_vira_episodio(self) -> None:
        episodes = find_alarm_episodes([False, True, True, True, False], persistence_window=3)
        assert len(episodes) == 1
        assert (episodes[0].start, episodes[0].length, episodes[0].end) == (1, 3, 3)

    def test_corrida_curta_nao_vira_episodio(self) -> None:
        assert find_alarm_episodes([False, True, True, False], persistence_window=3) == []

    def test_corridas_separadas_sao_episodios_distintos(self) -> None:
        episodes = find_alarm_episodes([True, True, False, True, True], persistence_window=2)
        assert [(e.start, e.length) for e in episodes] == [(0, 2), (3, 2)]

    def test_um_episodio_conta_uma_vez_por_mais_longo_que_seja(self) -> None:
        """O operador recebe um alarme, não vinte — é a unidade que importa."""
        episodes = find_alarm_episodes([True] * 20, persistence_window=3)
        assert len(episodes) == 1
        assert episodes[0].length == 20

    def test_rejeita_janela_invalida(self) -> None:
        with pytest.raises(InvalidValueError, match="persistência"):
            find_alarm_episodes([True], persistence_window=0)


class TestBacktest:
    def test_janela_maior_reduz_alarmes_falsos(self) -> None:
        """O argumento central a favor do parâmetro.

        Três picos isolados de ruído e uma falha real sustentada. Com janela 1,
        os quatro alarmam; com janela 3, só a falha real.
        """
        errors = _errors(5.0, 0.1, 5.0, 0.1, 5.0, 0.1, 5.0, 5.0, 5.0, 5.0)
        truth = [False, False, False, False, False, False, True, True, True, True]

        report = backtest_persistence(
            errors=errors,
            threshold=THRESHOLD,
            is_real_event=truth,
            persistence_windows=[1, 3],
        )

        assert report.outcome_for(1).episodes == 4
        assert report.outcome_for(3).episodes == 1
        assert report.false_alarms_avoided(baseline=1, candidate=3) == 3

    def test_janela_maior_nao_perde_a_deteccao_real(self) -> None:
        errors = _errors(5.0, 0.1, 5.0, 0.1, 5.0, 0.1, 5.0, 5.0, 5.0, 5.0)
        truth = [False, False, False, False, False, False, True, True, True, True]

        report = backtest_persistence(
            errors=errors,
            threshold=THRESHOLD,
            is_real_event=truth,
            persistence_windows=[1, 3],
        )
        assert report.outcome_for(3).recall == pytest.approx(1.0)
        assert report.detections_lost(baseline=1, candidate=3) == 0

    def test_janela_grande_demais_suprime_evento_real(self) -> None:
        """O custo do parâmetro, medido em vez de ignorado.

        Uma falha real de 4 janelas some quando se exige 8 consecutivas.
        """
        errors = _errors(0.1, 0.1, 5.0, 5.0, 5.0, 5.0, 0.1, 0.1)
        truth = [False, False, True, True, True, True, False, False]

        report = backtest_persistence(
            errors=errors,
            threshold=THRESHOLD,
            is_real_event=truth,
            persistence_windows=[1, 8],
        )
        assert report.outcome_for(8).episodes == 0
        assert report.detections_lost(baseline=1, candidate=8) == 4

    def test_precisao_melhora_com_a_janela(self) -> None:
        errors = _errors(5.0, 0.1, 5.0, 0.1, 5.0, 5.0, 5.0)
        truth = [False, False, False, False, True, True, True]

        report = backtest_persistence(
            errors=errors,
            threshold=THRESHOLD,
            is_real_event=truth,
            persistence_windows=[1, 3],
        )
        assert report.outcome_for(3).metrics.precision > report.outcome_for(1).metrics.precision

    def test_avalia_janelas_em_ordem_crescente_sem_duplicar(self) -> None:
        report = backtest_persistence(
            errors=_errors(5.0, 5.0, 5.0),
            threshold=THRESHOLD,
            is_real_event=[True, True, True],
            persistence_windows=[3, 1, 3],
        )
        assert [o.persistence_window for o in report.outcomes] == [1, 3]

    def test_relatorio_guarda_o_limiar_usado(self) -> None:
        report = backtest_persistence(
            errors=_errors(0.1),
            threshold=THRESHOLD,
            is_real_event=[False],
            persistence_windows=[1],
        )
        assert report.threshold == THRESHOLD
        assert report.evaluated_windows == 1

    def test_janela_nao_avaliada_falha_alto(self) -> None:
        report = backtest_persistence(
            errors=_errors(0.1),
            threshold=THRESHOLD,
            is_real_event=[False],
            persistence_windows=[1],
        )
        with pytest.raises(InvalidValueError, match="não avaliada"):
            report.outcome_for(99)

    def test_rejeita_rotulos_de_tamanho_diferente(self) -> None:
        with pytest.raises(InvalidValueError, match="mesmo tamanho"):
            backtest_persistence(
                errors=_errors(0.1, 0.2),
                threshold=THRESHOLD,
                is_real_event=[False],
                persistence_windows=[1],
            )

    def test_rejeita_historico_vazio(self) -> None:
        with pytest.raises(InsufficientDataError):
            backtest_persistence(
                errors=[],
                threshold=THRESHOLD,
                is_real_event=[],
                persistence_windows=[1],
            )
