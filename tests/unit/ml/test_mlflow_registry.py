"""Resolução e verificação de contrato no MLflow Registry.

Exercita toda a lógica de promoção e verificação com um cliente fake — sem subir
tracking server, sem rede. O que se testa aqui é justamente o que o v1 não
fazia: recusar servir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from eolica.infrastructure.featurestore import LagFeatureView
from eolica.infrastructure.ml.mlflow_registry import (
    TAG_FEATURE_NAMES,
    TAG_FEATURE_SIGNATURE,
    TAG_WINDOW_SIZE,
    MlflowModelRegistry,
)
from eolica.shared.errors import ConfigurationError, ModelUnavailableError

pytestmark = pytest.mark.ml

VIEW = LagFeatureView(features=("power", "wind_speed"), target="power", n_lags=6)
OTHER_VIEW = LagFeatureView(features=("power", "wind_speed"), target="power", n_lags=12)


@dataclass
class FakeVersion:
    version: str = "3"
    run_id: str = "abc123"
    tags: dict[str, str] = field(default_factory=dict)


class FakeClient:
    """Cliente MLflow em memória, indexado por (nome, alias)."""

    def __init__(self, versions: dict[tuple[str, str], FakeVersion] | None = None) -> None:
        self._versions = versions or {}
        self.promotions: list[tuple[str, str, str]] = []

    def get_model_version_by_alias(self, *, name: str, alias: str) -> FakeVersion:
        try:
            return self._versions[(name, alias)]
        except KeyError:
            raise RuntimeError(f"alias '{alias}' não registrado para '{name}'") from None

    def set_registered_model_alias(self, *, name: str, alias: str, version: str) -> None:
        self.promotions.append((name, alias, version))


def _registry(
    versions: dict[tuple[str, str], FakeVersion] | None = None,
) -> MlflowModelRegistry:
    mapping: dict[tuple[str, str], FakeVersion] = {
        ("wind-power-forecaster", "production"): FakeVersion(
            tags={
                TAG_FEATURE_SIGNATURE: VIEW.signature,
                TAG_WINDOW_SIZE: "60",
                TAG_FEATURE_NAMES: json.dumps(["power", "wind_speed"]),
            }
        )
    }
    mapping.update(versions or {})
    return MlflowModelRegistry(tracking_uri="memory://", client=FakeClient(mapping))


class TestRecusaDeAliasNaoPromovido:
    @pytest.mark.parametrize("alias", ["latest", "Latest", "LATEST", "none", ""])
    def test_recusa_aliases_nao_serviveis(self, alias: str) -> None:
        """O v1 servia `models:/{nome}/latest` — qualquer experimento registrado
        num notebook virava o modelo de produção sem review."""
        with pytest.raises(ConfigurationError, match="não é um alias servível"):
            _registry().resolve("wind-power-forecaster", alias=alias)

    def test_mensagem_orienta_a_promocao_consciente(self) -> None:
        with pytest.raises(ConfigurationError, match="conscientemente"):
            _registry().resolve("wind-power-forecaster", alias="latest")

    def test_recusa_promover_para_alias_proibido(self) -> None:
        with pytest.raises(ConfigurationError, match="promovível"):
            _registry().promote("wind-power-forecaster", version="3", alias="latest")


class TestResolucao:
    def test_resolve_alias_promovido(self) -> None:
        model = _registry().resolve("wind-power-forecaster", alias="production")
        assert model.version == "3"
        assert model.run_id == "abc123"

    def test_alias_e_normalizado_para_minusculas(self) -> None:
        model = _registry().resolve("wind-power-forecaster", alias="Production")
        assert model.alias == "production"

    def test_identidade_e_rastreavel(self) -> None:
        """`model_version` na previsão precisa apontar para algo reproduzível."""
        model = _registry().resolve("wind-power-forecaster", alias="production")
        assert model.identity == "wind-power-forecaster@production#3"

    def test_alias_inexistente_e_modelo_indisponivel(self) -> None:
        """503, não 500: é transitório e o cliente deve tentar de novo."""
        with pytest.raises(ModelUnavailableError, match="staging"):
            _registry().resolve("wind-power-forecaster", alias="staging")

    def test_le_os_metadados_gravados_no_treino(self) -> None:
        model = _registry().resolve("wind-power-forecaster", alias="production")
        assert model.window_size == 60
        assert model.feature_names == ("power", "wind_speed")


class TestVerificacaoDoContratoDeFeatures:
    def test_assinatura_igual_passa(self) -> None:
        registry = _registry()
        model = registry.resolve("wind-power-forecaster", alias="production")
        registry.verify_feature_contract(model, VIEW)

    def test_assinatura_divergente_recusa_servir(self) -> None:
        """O bug silencioso do v1 vira falha alta.

        Mudar n_lags de 6 para 12 sem retreinar passava despercebido: o modelo
        recebia features com significado diferente e só devolvia previsão pior.
        """
        registry = _registry()
        model = registry.resolve("wind-power-forecaster", alias="production")
        with pytest.raises(ConfigurationError, match="não corresponde à do treino"):
            registry.verify_feature_contract(model, OTHER_VIEW)

    def test_erro_mostra_as_duas_assinaturas(self) -> None:
        registry = _registry()
        model = registry.resolve("wind-power-forecaster", alias="production")
        with pytest.raises(ConfigurationError) as exc:
            registry.verify_feature_contract(model, OTHER_VIEW)
        assert exc.value.context["trained_with"] == VIEW.signature
        assert exc.value.context["serving_with"] == OTHER_VIEW.signature

    def test_modelo_sem_assinatura_nao_e_servivel(self) -> None:
        """Artefato treinado fora do pipeline não entra em produção."""
        registry = _registry({("wind-power-forecaster", "production"): FakeVersion(tags={})})
        model = registry.resolve("wind-power-forecaster", alias="production")
        with pytest.raises(ConfigurationError, match="não registrou a assinatura"):
            registry.verify_feature_contract(model, VIEW)


class TestPromocao:
    def test_promocao_aponta_o_alias_para_a_versao(self) -> None:
        client = FakeClient({})
        registry = MlflowModelRegistry(tracking_uri="memory://", client=client)
        registry.promote("wind-power-forecaster", version="7", alias="Production")
        assert client.promotions == [("wind-power-forecaster", "production", "7")]
