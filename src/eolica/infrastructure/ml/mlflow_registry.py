"""Adaptador do MLflow Model Registry. Requer o extra `[ml]`.

Fecha o ciclo que o v1 deixava aberto: um modelo treinado num notebook virava o
modelo de produção sem nenhuma verificação.

Três recusas explícitas, todas na subida do processo:

1. **`latest` não é um alias servível.** O v1 carregava
   `models:/{nome}/latest`, que resolve para a versão mais recente *registrada*.
   Qualquer `mlflow.register_model()` rodado num notebook às duas da manhã
   passava a ser o que a API servia — sem review, sem promoção, sem aviso.

2. **Assinatura de feature view divergente é erro, não degradação.** A versão
   registrada carrega a assinatura do conjunto de features com que foi treinada.
   Se a view em uso não bater, o carregamento falha em vez de servir um modelo
   com features que significam outra coisa.

3. **Modelo sem os metadados exigidos não é servível.** Um modelo sem
   `window_size` ou sem lista de features registrada é indistinguível de um
   artefato solto num diretório.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eolica.shared.errors import ConfigurationError, ModelUnavailableError

if TYPE_CHECKING:
    from eolica.infrastructure.featurestore import FeatureView

# Chaves de tag gravadas no registro. O prefixo evita colisão com tags do
# próprio MLflow ou de outras equipes no mesmo tracking server.
TAG_FEATURE_SIGNATURE = "eolica.feature_view_signature"
TAG_FEATURE_NAMES = "eolica.feature_names"
TAG_WINDOW_SIZE = "eolica.window_size"
TAG_BEATS_BASELINE = "eolica.beats_baseline"

FORBIDDEN_ALIASES = frozenset({"latest", "none", ""})


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """Uma versão resolvida no registry, com o que é preciso para auditá-la."""

    name: str
    version: str
    run_id: str
    alias: str
    tags: dict[str, str]

    @property
    def identity(self) -> str:
        """Identidade legível, usada como `model_version` nas previsões."""
        return f"{self.name}@{self.alias}#{self.version}"

    @property
    def feature_signature(self) -> str | None:
        return self.tags.get(TAG_FEATURE_SIGNATURE)

    @property
    def window_size(self) -> int | None:
        raw = self.tags.get(TAG_WINDOW_SIZE)
        return int(raw) if raw is not None else None

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        raw = self.tags.get(TAG_FEATURE_NAMES)
        if raw is None:
            return None
        return tuple(json.loads(raw))


class MlflowModelRegistry:
    """Resolve e carrega modelos promovidos, verificando o contrato de features.

    O cliente é injetável para que os testes exercitem toda a lógica de
    resolução e verificação sem subir um tracking server.
    """

    def __init__(self, *, tracking_uri: str, client: Any | None = None) -> None:
        self._tracking_uri = tracking_uri
        self._client = client or self._default_client(tracking_uri)

    @staticmethod
    def _default_client(tracking_uri: str) -> Any:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError as exc:  # pragma: no cover - depende do extra [ml]
            raise ConfigurationError(
                "MLflow não está instalado. Instale o extra: pip install '.[ml]'"
            ) from exc
        mlflow.set_tracking_uri(tracking_uri)
        return MlflowClient(tracking_uri=tracking_uri)

    # ── resolução ────────────────────────────────────────────────────────────

    def resolve(self, name: str, *, alias: str) -> RegisteredModel:
        """Resolve um alias promovido para uma versão concreta.

        Usa aliases e não *stages*: o MLflow depreciou stages na versão 2.9, e o
        v1 já usava a API antiga (`get_latest_versions(stages=["None"])`) que
        emite aviso de depreciação desde então.
        """
        normalised = alias.strip().lower()
        if normalised in FORBIDDEN_ALIASES:
            raise ConfigurationError(
                f"'{alias}' não é um alias servível. Promova uma versão "
                "conscientemente (ex.: 'production') em vez de servir a mais recente.",
                model=name,
                alias=alias,
            )

        try:
            version = self._client.get_model_version_by_alias(name=name, alias=normalised)
        except Exception as exc:
            raise ModelUnavailableError(
                name, f"não há versão com o alias '{normalised}' no registry ({exc})"
            ) from exc

        return RegisteredModel(
            name=name,
            version=str(version.version),
            run_id=str(version.run_id),
            alias=normalised,
            tags=dict(getattr(version, "tags", {}) or {}),
        )

    def verify_feature_contract(self, model: RegisteredModel, view: FeatureView) -> None:
        """Recusa servir se a view em uso divergir da usada no treino.

        É o que transforma o bug silencioso do v1 — mudar `n_lags` e continuar
        servindo, com previsão pior e nenhum sinal — em falha no readiness probe.
        """
        recorded = model.feature_signature
        if recorded is None:
            raise ConfigurationError(
                f"A versão {model.identity} não registrou a assinatura da feature view. "
                "Modelos treinados fora do pipeline não são servíveis.",
                model=model.name,
                version=model.version,
            )
        if recorded != view.signature:
            raise ConfigurationError(
                "A feature view em uso não corresponde à do treino. "
                "Retreine o modelo ou restaure a configuração de features.",
                model=model.identity,
                trained_with=recorded,
                serving_with=view.signature,
            )

    # ── carregamento ─────────────────────────────────────────────────────────

    def load_forecaster(self, name: str, *, alias: str, view: FeatureView) -> Any:
        """Carrega o previsor promovido, já amarrado à feature view verificada."""
        from eolica.infrastructure.ml.xgboost_forecaster import XGBoostPowerForecaster

        model = self.resolve(name, alias=alias)
        self.verify_feature_contract(model, view)

        try:
            import mlflow.xgboost

            booster = mlflow.xgboost.load_model(f"models:/{name}@{alias}")
        except Exception as exc:
            raise ModelUnavailableError(name, f"falha ao carregar o artefato: {exc}") from exc

        return XGBoostPowerForecaster(
            booster=booster,
            view=view,
            version=model.identity,
            trained_signature=model.feature_signature,
        )

    def load_health_model(self, name: str, *, alias: str) -> Any:
        """Carrega o detector promovido, com scaler e janela vindos das tags."""
        from eolica.infrastructure.ml.torch_autoencoder import (
            StandardScaler,
            TorchReconstructionModel,
        )

        model = self.resolve(name, alias=alias)
        window_size = model.window_size
        feature_names = model.feature_names
        if window_size is None or feature_names is None:
            raise ConfigurationError(
                f"A versão {model.identity} não registrou window_size e feature_names. "
                "Sem esses metadados o modelo não é servível.",
                model=model.name,
                version=model.version,
            )

        try:
            import mlflow.pytorch

            network = mlflow.pytorch.load_model(f"models:/{name}@{alias}")
            scaler_payload = self._client.download_artifacts(model.run_id, "scaler.json")
        except Exception as exc:
            raise ModelUnavailableError(name, f"falha ao carregar o artefato: {exc}") from exc

        scaler = StandardScaler.from_dict(
            json.loads(Path(scaler_payload).read_text(encoding="utf-8"))
        )

        return TorchReconstructionModel(
            model=network,
            scaler=scaler,
            window_size=window_size,
            feature_names=feature_names,
            version=model.identity,
        )

    # ── promoção ─────────────────────────────────────────────────────────────

    def promote(self, name: str, *, version: str, alias: str) -> None:
        """Aponta um alias para uma versão. É o ato de "colocar em produção".

        Separado do treino de propósito: registrar é automático, promover é
        decisão humana.
        """
        normalised = alias.strip().lower()
        if normalised in FORBIDDEN_ALIASES:
            raise ConfigurationError(f"'{alias}' não é um alias promovível", model=name)
        self._client.set_registered_model_alias(name=name, alias=normalised, version=version)
