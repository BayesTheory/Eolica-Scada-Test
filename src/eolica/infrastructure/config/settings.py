"""Configuração tipada, validada na subida do processo.

Substitui três fontes de verdade concorrentes do v1:

- `constants.py`, com `FORECASTING_EXPERIMENT_NAME = "Wind Power Forecasting"`;
- `config.yaml`, com `experiment_name: "Power Forecasting"` — nome *diferente*
  para o mesmo experimento;
- literais espalhados pelo código (`FORECASTING_MODEL_NAME = 'wind-power-forecaster'`
  no meio do `inference_api.py`).

E, principalmente, substitui `os.environ['GOOGLE_API_KEY'] = "AIza..."` no
código-fonte. Aqui segredo é `SecretStr`, vem só do ambiente, não tem default e
não aparece em `repr()` nem em log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Configuração de runtime, vinda do ambiente (prefixo `EOLICA_`).

    Tudo que muda entre dev/staging/prod mora aqui. O que descreve um
    *experimento* de ML (features, hiperparâmetros) mora em `configs/*.yaml`,
    porque é versionado junto do código e revisado em PR.
    """

    model_config = SettingsConfigDict(
        env_prefix="EOLICA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    # ── ambiente ─────────────────────────────────────────────────────────────
    environment: Literal["local", "ci", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # ── dados ────────────────────────────────────────────────────────────────
    data_path: Path = PROJECT_ROOT / "data" / "processed" / "scada_resampled_10min_base.csv"
    sample_data_path: Path = PROJECT_ROOT / "data" / "samples" / "scada_sample.csv"
    sampling_interval_minutes: int = Field(default=10, ge=1)

    # ── registry de modelos ──────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    health_model_name: str = "wind-turbine-health-specialist"
    forecast_model_name: str = "wind-power-forecaster"
    model_stage: str = "Production"
    """Alias do registry a servir. `Production` — nunca `latest`.

    O v1 usava `models:/{nome}/latest`, que resolve para a versão mais recente
    *registrada*, incluindo a que alguém acabou de treinar num notebook. Um
    experimento local podia virar o modelo de produção sem review.
    """

    # ── detector de anomalia ─────────────────────────────────────────────────
    anomaly_threshold_percentile: float = Field(default=99.5, ge=0.0, le=100.0)
    persistence_window: int = Field(default=6, ge=1)
    """Janelas consecutivas acima do limiar para caracterizar alerta.

    6 janelas de 10 minutos = 1 hora de desvio sustentado. Estava no
    `config.yaml` do v1 e nenhuma linha de código o lia.
    """

    health_window_size: int = Field(default=60, ge=2)

    # ── previsão ─────────────────────────────────────────────────────────────
    forecast_n_lags: int = Field(default=6, ge=1)
    forecast_horizon_steps: int = Field(default=1, ge=1)

    # ── drift ────────────────────────────────────────────────────────────────
    drift_bins: int = Field(default=10, ge=2)
    drift_reference_days: int = Field(default=30, ge=1)

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"  # noqa: S104 — em container, o bind precisa ser amplo
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_allowed_origins: tuple[str, ...] = ()
    """Vazio por padrão — e deve continuar vazio.

    O frontend é servido pela própria API, na mesma origem. Só preencha se
    existir um cliente hospedado em outro domínio.
    """

    enable_metrics: bool = True
    serve_frontend: bool = True
    """Serve `frontend/dist` quando ele existe. Desligue no container de treino."""

    # ── co-piloto (opcional, desligado por padrão) ───────────────────────────
    copilot_enabled: bool = False
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.0-flash"

    @field_validator("data_path", "sample_data_path")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser()

    @model_validator(mode="after")
    def _copilot_requires_key(self) -> Settings:
        """Falha na subida, não na primeira pergunta do operador."""
        if self.copilot_enabled and self.gemini_api_key is None:
            raise ValueError(
                "EOLICA_COPILOT_ENABLED=true exige EOLICA_GEMINI_API_KEY. "
                "Defina no .env (nunca no código)."
            )
        return self

    @model_validator(mode="after")
    def _production_forbids_latest_stage(self) -> Settings:
        """Em produção, servir `latest` é proibido explicitamente."""
        if self.environment == "production" and self.model_stage.lower() == "latest":
            raise ValueError(
                "EOLICA_MODEL_STAGE='latest' não é permitido em produção: "
                "use um alias promovido conscientemente (ex.: 'Production')."
            )
        return self
