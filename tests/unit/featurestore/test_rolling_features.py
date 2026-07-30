"""Features de janela móvel: desvio padrão e taxa de variação.

A análise exploratória da v1 (`testes/analyze_features.py`) rankeava
`GeneratorTemperature_std_1h` e `*_roc_1h` entre as features mais preditivas para
o alvo de pré-falha — usando ANOVA, informação mútua, importância do XGBoost e
RFE. Mas o script apontava para um CSV que o pipeline nunca gerava
(`scada_resampled_10min_features.csv`), então a conclusão nunca virou código.

O risco desta classe de feature é vazamento temporal: `rolling(6).std()` numa
série inclui o instante atual, e o instante atual é justamente o que se quer
prever. Um modelo treinado assim tem métrica offline excelente e é inútil em
produção, porque na hora da decisão aquele valor ainda não existe.

Por isso `TestCausalidade` é o teste central deste arquivo.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eolica.infrastructure.featurestore import (
    CompositeFeatureView,
    LagFeatureView,
    RollingFeatureView,
)
from eolica.shared.errors import ConfigurationError, InsufficientDataError

WINDOW_1H = 6  # 6 passos de 10 minutos


@pytest.fixture
def frame() -> pd.DataFrame:
    index = pd.date_range("2022-01-14", periods=30, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "power": [float(i) for i in range(30)],
            "generator_temperature": [40.0 + (i % 5) for i in range(30)],
        },
        index=index,
    )


@pytest.fixture
def view() -> RollingFeatureView:
    return RollingFeatureView(
        features=("power", "generator_temperature"), target="power", window_steps=WINDOW_1H
    )


class TestContrato:
    def test_gera_desvio_e_taxa_de_variacao_por_feature(self, view: RollingFeatureView) -> None:
        assert view.feature_names == (
            "generator_temperature_roc_6",
            "generator_temperature_std_6",
            "power_roc_6",
            "power_std_6",
        )

    def test_exige_um_passo_a_mais_que_a_janela(self, view: RollingFeatureView) -> None:
        """A taxa de variação compara o início e o fim da janela: precisa de w+1
        observações, não w."""
        assert view.required_history == WINDOW_1H + 1

    def test_assinatura_muda_com_o_tamanho_da_janela(self, view: RollingFeatureView) -> None:
        other = RollingFeatureView(
            features=("power", "generator_temperature"), target="power", window_steps=12
        )
        assert view.signature != other.signature

    def test_rejeita_janela_menor_que_dois(self) -> None:
        with pytest.raises(ConfigurationError, match="window_steps"):
            RollingFeatureView(features=("power",), target="power", window_steps=1)


class TestCausalidade:
    """Nenhuma feature pode conhecer o instante que está tentando prever."""

    def test_desvio_nao_inclui_o_instante_alvo(self, frame: pd.DataFrame) -> None:
        """`rolling(6).std()` cru incluiria x[t]. Aqui a série é deslocada antes."""
        view = RollingFeatureView(features=("power",), target="power", window_steps=WINDOW_1H)
        features, _ = view.build_training_matrix(frame)

        target_time = frame.index[10]
        expected = frame["power"].iloc[4:10].std(ddof=0)
        assert features.loc[target_time, "power_std_6"] == pytest.approx(expected)

    def test_taxa_de_variacao_nao_inclui_o_instante_alvo(self, frame: pd.DataFrame) -> None:
        view = RollingFeatureView(features=("power",), target="power", window_steps=WINDOW_1H)
        features, _ = view.build_training_matrix(frame)

        target_time = frame.index[10]
        expected = (frame["power"].iloc[9] - frame["power"].iloc[3]) / WINDOW_1H
        assert features.loc[target_time, "power_roc_6"] == pytest.approx(expected)

    def test_alterar_o_futuro_nao_muda_nenhuma_feature_do_passado(
        self, frame: pd.DataFrame
    ) -> None:
        """O teste decisivo contra vazamento.

        Se reescrever todos os valores a partir de t mudar qualquer feature cujo
        alvo é anterior a t, há informação do futuro entrando na matriz.
        """
        view = RollingFeatureView(
            features=("power", "generator_temperature"), target="power", window_steps=WINDOW_1H
        )
        original, _ = view.build_training_matrix(frame)

        tampered = frame.copy()
        tampered.iloc[15:] = 999.0
        after, _ = view.build_training_matrix(tampered)

        untouched_targets = original.index[original.index < frame.index[15]]
        pd.testing.assert_frame_equal(original.loc[untouched_targets], after.loc[untouched_targets])


class TestAusenciaDeSkew:
    def test_treino_e_serving_produzem_o_mesmo_vetor(
        self, frame: pd.DataFrame, view: RollingFeatureView
    ) -> None:
        features, _ = view.build_training_matrix(frame)
        target_time = frame.index[20]
        serving = view.build_inference_vector(frame.loc[: frame.index[19]])
        pd.testing.assert_series_equal(
            features.loc[target_time], serving.iloc[0], check_names=False
        )

    @pytest.mark.parametrize("position", [8, 15, 22, 29])
    def test_coincidem_em_qualquer_ponto_da_serie(
        self, frame: pd.DataFrame, view: RollingFeatureView, position: int
    ) -> None:
        features, _ = view.build_training_matrix(frame)
        serving = view.build_inference_vector(frame.loc[: frame.index[position - 1]])
        pd.testing.assert_series_equal(
            features.loc[frame.index[position]], serving.iloc[0], check_names=False
        )

    def test_historico_curto_e_recusado(self, view: RollingFeatureView) -> None:
        short = pd.DataFrame(
            {"power": [1.0, 2.0], "generator_temperature": [40.0, 41.0]},
            index=pd.date_range("2022-01-14", periods=2, freq="10min", tz="UTC"),
        )
        with pytest.raises(InsufficientDataError):
            view.build_inference_vector(short)


class TestComposicao:
    """Lag e janela móvel combinados numa view só."""

    @pytest.fixture
    def composite(self) -> CompositeFeatureView:
        return CompositeFeatureView(
            views=(
                LagFeatureView(features=("power",), target="power", n_lags=3),
                RollingFeatureView(features=("power",), target="power", window_steps=WINDOW_1H),
            ),
            target="power",
        )

    def test_reune_as_colunas_das_duas_views(self, composite: CompositeFeatureView) -> None:
        assert composite.feature_names == (
            "power_lag_1",
            "power_lag_2",
            "power_lag_3",
            "power_roc_6",
            "power_std_6",
        )

    def test_exige_o_maior_historico_entre_as_views(self, composite: CompositeFeatureView) -> None:
        """3 lags exigem 3 observações; a janela de 1h exige 7. Vale a maior."""
        assert composite.required_history == WINDOW_1H + 1

    def test_assinatura_deriva_das_views_componentes(self, composite: CompositeFeatureView) -> None:
        different = CompositeFeatureView(
            views=(
                LagFeatureView(features=("power",), target="power", n_lags=6),
                RollingFeatureView(features=("power",), target="power", window_steps=WINDOW_1H),
            ),
            target="power",
        )
        assert composite.signature != different.signature

    def test_mantem_a_ausencia_de_skew(
        self, frame: pd.DataFrame, composite: CompositeFeatureView
    ) -> None:
        features, _ = composite.build_training_matrix(frame)
        target_time = frame.index[20]
        serving = composite.build_inference_vector(frame.loc[: frame.index[19]])
        pd.testing.assert_series_equal(
            features.loc[target_time], serving.iloc[0], check_names=False
        )

    def test_rejeita_composicao_vazia(self) -> None:
        with pytest.raises(ConfigurationError, match="ao menos uma view"):
            CompositeFeatureView(views=(), target="power")

    def test_rejeita_views_com_alvos_divergentes(self) -> None:
        with pytest.raises(ConfigurationError, match="alvo"):
            CompositeFeatureView(
                views=(
                    LagFeatureView(features=("power",), target="power", n_lags=3),
                    RollingFeatureView(
                        features=("power",), target="wind_speed", window_steps=WINDOW_1H
                    ),
                ),
                target="power",
            )
