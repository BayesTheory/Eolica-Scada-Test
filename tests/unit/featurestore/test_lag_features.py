"""Feature store: fonte única de verdade das features tabulares.

Este arquivo existe por causa de um bug específico do v1.

As features de lag do XGBoost eram construídas em **dois lugares diferentes**:

- no treino, `train_models._create_lagged_features_for_trees()` usava
  `df[col].shift(lag)` sobre o DataFrame inteiro;
- no serving, `Forecaster.predict_next_step()` montava o vetor à mão com
  `last_window[col].iloc[-lag]`.

Duas implementações da mesma ideia, em arquivos diferentes, sem nenhum teste
ligando uma à outra. Pior: o `n_lags` vinha de caminhos distintos do
`config.yaml` — `model_params['params']['n_lags']` no treino e
`forecasting_params['n_lags']` no serving. Nenhuma das duas chaves existia no
arquivo, então ambas caíam no default `6` e o sistema funcionava *por acidente*.
Bastava alguém adicionar `n_lags: 12` na seção que parecia certa para o modelo
passar a receber, em produção, features com significado diferente das do treino
— sem erro, sem log, só previsão pior.

O teste que importa aqui é `test_treino_e_serving_produzem_o_mesmo_vetor`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eolica.infrastructure.featurestore import LagFeatureView
from eolica.shared.errors import ContractViolationError, InsufficientDataError


@pytest.fixture
def frame() -> pd.DataFrame:
    """Série sintética curta, com valores distintos por coluna e por instante."""
    index = pd.date_range("2022-01-14", periods=20, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "power": [float(i) for i in range(20)],
            "wind_speed": [float(i) * 0.5 for i in range(20)],
        },
        index=index,
    )


@pytest.fixture
def view() -> LagFeatureView:
    return LagFeatureView(features=("power", "wind_speed"), target="power", n_lags=3)


class TestContratoDaView:
    def test_nomes_das_colunas_sao_deterministicos(self, view: LagFeatureView) -> None:
        assert view.feature_names == (
            "power_lag_1",
            "power_lag_2",
            "power_lag_3",
            "wind_speed_lag_1",
            "wind_speed_lag_2",
            "wind_speed_lag_3",
        )

    def test_a_ordem_das_colunas_e_estavel_entre_instancias(self) -> None:
        """A ordem é o contrato: XGBoost casa features por posição.

        O v1 dependia de `model.feature_names_in_` para reordenar em serving —
        o que funciona, mas só porque o sklearn guarda os nomes. Um export para
        ONNX ou um `Booster` cru perderia essa rede de segurança.
        """
        first = LagFeatureView(features=("power", "wind_speed"), target="power", n_lags=2)
        second = LagFeatureView(features=("wind_speed", "power"), target="power", n_lags=2)
        assert first.feature_names == second.feature_names

    def test_a_view_tem_uma_assinatura_versionada(self, view: LagFeatureView) -> None:
        """Identidade estável do conjunto de features, para gravar junto do modelo."""
        assert (
            view.signature
            == LagFeatureView(features=("power", "wind_speed"), target="power", n_lags=3).signature
        )

    def test_mudar_n_lags_muda_a_assinatura(self, view: LagFeatureView) -> None:
        """É isto que teria pego o bug: mudar n_lags muda a identidade da view,
        e um modelo treinado com a antiga recusa a servir com a nova."""
        other = LagFeatureView(features=("power", "wind_speed"), target="power", n_lags=6)
        assert view.signature != other.signature

    def test_rejeita_n_lags_invalido(self) -> None:
        with pytest.raises(ValueError, match="n_lags"):
            LagFeatureView(features=("power",), target="power", n_lags=0)

    def test_rejeita_lista_de_features_vazia(self) -> None:
        with pytest.raises(ValueError, match="features"):
            LagFeatureView(features=(), target="power", n_lags=3)


class TestMatrizDeTreino:
    def test_descarta_as_linhas_sem_historico_suficiente(
        self, frame: pd.DataFrame, view: LagFeatureView
    ) -> None:
        features, target = view.build_training_matrix(frame)
        assert len(features) == len(frame) - 3
        assert len(target) == len(features)

    def test_lag_1_e_o_valor_do_instante_anterior(
        self, frame: pd.DataFrame, view: LagFeatureView
    ) -> None:
        features, target = view.build_training_matrix(frame)
        row = features.iloc[0]
        assert row["power_lag_1"] == 2.0
        assert row["power_lag_2"] == 1.0
        assert row["power_lag_3"] == 0.0
        assert target.iloc[0] == 3.0

    def test_colunas_saem_na_ordem_do_contrato(
        self, frame: pd.DataFrame, view: LagFeatureView
    ) -> None:
        features, _ = view.build_training_matrix(frame)
        assert tuple(features.columns) == view.feature_names

    def test_exige_as_colunas_declaradas(self, view: LagFeatureView) -> None:
        with pytest.raises(ContractViolationError, match="wind_speed"):
            view.build_training_matrix(pd.DataFrame({"power": [1.0, 2.0, 3.0, 4.0, 5.0]}))


class TestVetorDeInferencia:
    def test_produz_uma_unica_linha(self, frame: pd.DataFrame, view: LagFeatureView) -> None:
        vector = view.build_inference_vector(frame)
        assert vector.shape == (1, len(view.feature_names))

    def test_lag_1_e_a_observacao_mais_recente(
        self, frame: pd.DataFrame, view: LagFeatureView
    ) -> None:
        vector = view.build_inference_vector(frame)
        assert vector.iloc[0]["power_lag_1"] == 19.0
        assert vector.iloc[0]["power_lag_2"] == 18.0

    def test_recusa_historico_curto_em_vez_de_preencher_com_nan(self, view: LagFeatureView) -> None:
        """O v1 levantava ValueError aqui, o que estava certo — mas o cálculo
        do mínimo (`len(df) < n_lags`) era feito com um `n_lags` que podia não
        ser o do treino."""
        short = pd.DataFrame(
            {"power": [1.0, 2.0], "wind_speed": [0.5, 1.0]},
            index=pd.date_range("2022-01-14", periods=2, freq="10min", tz="UTC"),
        )
        with pytest.raises(InsufficientDataError):
            view.build_inference_vector(short)


class TestAusenciaDeSkew:
    """O teste que o v1 não tinha e que teria evitado toda a classe de bug."""

    def test_treino_e_serving_produzem_o_mesmo_vetor(
        self, frame: pd.DataFrame, view: LagFeatureView
    ) -> None:
        """Para o mesmo instante alvo, as duas rotas têm que coincidir bit a bit.

        Rota de treino: a linha da matriz cujo alvo é `t`.
        Rota de serving: o vetor construído com o histórico que termina em `t-1`.

        Se alguém reintroduzir uma segunda implementação de lag, este teste cai.
        """
        features, _ = view.build_training_matrix(frame)

        target_time = frame.index[10]
        training_row = features.loc[target_time]
        serving_row = view.build_inference_vector(frame.loc[: frame.index[9]]).iloc[0]

        pd.testing.assert_series_equal(
            training_row, serving_row, check_names=False, check_dtype=True
        )

    @pytest.mark.parametrize("position", [5, 9, 14, 19])
    def test_coincidem_em_qualquer_ponto_da_serie(
        self, frame: pd.DataFrame, view: LagFeatureView, position: int
    ) -> None:
        features, _ = view.build_training_matrix(frame)
        target_time = frame.index[position]
        history = frame.loc[: frame.index[position - 1]]
        pd.testing.assert_series_equal(
            features.loc[target_time],
            view.build_inference_vector(history).iloc[0],
            check_names=False,
        )
