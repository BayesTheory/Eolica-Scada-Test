"""Adaptadores reais de ML. Exigem o extra `[ml]`; pulam sem ele."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta

import pytest

from eolica.domain.turbine import (
    OperatingStatus,
    PitchAngle,
    PowerKw,
    ReadingWindow,
    RotorSpeed,
    Temperature,
    TurbineReading,
    WindSpeed,
)

torch = pytest.importorskip("torch", reason="requer o extra [ml]")

from eolica.infrastructure.ml.torch_autoencoder import (  # noqa: E402
    LSTMAutoencoder,
    StandardScaler,
    TorchReconstructionModel,
)

pytestmark = pytest.mark.ml

STEP = timedelta(minutes=10)
FEATURES = ("generator_temperature", "pitch", "power", "rotor_speed", "wind_speed")


def _window(count: int, *, temperature: float = 40.0) -> ReadingWindow:
    base = datetime(2022, 1, 14, tzinfo=UTC)
    return ReadingWindow.of(
        [
            TurbineReading(
                timestamp=base + i * STEP,
                wind_speed=WindSpeed(5.0),
                power=PowerKw(1.0),
                rotor_speed=RotorSpeed(30.0),
                generator_temperature=Temperature(temperature),
                pitch=PitchAngle(20.0),
                status=OperatingStatus.PRODUCING,
            )
            for i in range(count)
        ],
        expected_interval=STEP,
    )


@pytest.fixture
def adapter() -> TorchReconstructionModel:
    torch.manual_seed(0)
    return TorchReconstructionModel(
        model=LSTMAutoencoder(n_features=5, hidden_size=16, n_layers=1),
        scaler=StandardScaler(means=[40.0, 20.0, 1.0, 30.0, 5.0], deviations=[5.0] * 5),
        window_size=6,
        feature_names=FEATURES,
    )


class TestLSTMAutoencoder:
    def test_reconstrucao_preserva_o_shape_da_entrada(self) -> None:
        model = LSTMAutoencoder(n_features=5, hidden_size=16, n_layers=1)
        x = torch.randn(4, 12, 5)
        assert model(x).shape == x.shape

    def test_e_deterministico_em_modo_de_avaliacao(self) -> None:
        """Diferente da geração autorregressiva da v1, não há acúmulo de erro:
        a mesma entrada dá exatamente a mesma saída."""
        model = LSTMAutoencoder(n_features=5, hidden_size=16, n_layers=1).eval()
        x = torch.randn(2, 10, 5)
        with torch.no_grad():
            assert torch.allclose(model(x), model(x))

    def test_erro_por_janela_e_nao_negativo(self) -> None:
        model = LSTMAutoencoder(n_features=5, hidden_size=16, n_layers=1).eval()
        assert (model.window_errors(torch.randn(8, 10, 5)) >= 0).all()

    def test_dropout_com_camada_unica_nao_emite_warning(self) -> None:
        """PyTorch avisa quando `dropout` é passado com `num_layers=1`.

        Zerar o dropout explicitamente nesse caso mantém o log limpo e deixa a
        intenção no código em vez de num warning silenciado.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            LSTMAutoencoder(n_features=3, hidden_size=8, n_layers=1, dropout=0.5)


class TestStandardScaler:
    def test_padroniza_para_media_zero(self) -> None:
        matrix = torch.tensor([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        scaled = StandardScaler.fit(matrix).transform(matrix)
        assert torch.allclose(scaled.mean(dim=0), torch.zeros(2), atol=1e-6)

    def test_feature_constante_nao_divide_por_zero(self) -> None:
        """Sensor travado: `PitchDeg` fica em 14.034 por longos períodos."""
        matrix = torch.tensor([[5.0], [5.0], [5.0]])
        scaled = StandardScaler.fit(matrix).transform(matrix)
        assert torch.isfinite(scaled).all()

    def test_serializa_sem_pickle(self) -> None:
        """A v1 gravava o scaler do sklearn com pickle — formato que amarra o
        artefato à versão da lib e executa código no carregamento."""
        original = StandardScaler(means=[1.0, 2.0], deviations=[3.0, 4.0])
        restored = StandardScaler.from_dict(original.to_dict())
        assert restored.means == original.means
        assert restored.deviations == original.deviations


class TestAdaptador:
    def test_satisfaz_a_porta_do_dominio(self, adapter: TorchReconstructionModel) -> None:
        from eolica.domain.health import ReconstructionModel

        assert isinstance(adapter, ReconstructionModel)

    def test_produz_um_erro_por_sub_janela(self, adapter: TorchReconstructionModel) -> None:
        errors = adapter.reconstruction_errors(_window(20))
        assert len(errors) == 20 - 6 + 1

    def test_janela_curta_demais_nao_produz_erro(self, adapter: TorchReconstructionModel) -> None:
        assert adapter.reconstruction_errors(_window(3)) == []

    def test_desvio_grande_eleva_o_erro_de_reconstrucao(
        self, adapter: TorchReconstructionModel
    ) -> None:
        """Sanidade: dado longe da distribuição de calibração reconstrói pior."""
        normal = adapter.reconstruction_errors(_window(20, temperature=40.0))
        anomalous = adapter.reconstruction_errors(_window(20, temperature=400.0))
        assert max(e.value for e in anomalous) > max(e.value for e in normal)

    def test_recusa_features_incompativeis_com_o_modelo(self) -> None:
        with pytest.raises(Exception, match="features"):
            TorchReconstructionModel(
                model=LSTMAutoencoder(n_features=5, hidden_size=8, n_layers=1),
                scaler=StandardScaler(means=[0.0] * 3, deviations=[1.0] * 3),
                window_size=6,
                feature_names=("power", "wind_speed", "pitch"),
            )
