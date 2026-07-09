"""Contratos de request e response da API.

Modelos Pydantic explícitos, com exemplos, para que o OpenAPI gerado seja
documentação de verdade. O v1 devolvia um `dict` montado à mão dentro do
handler: o `/docs` mostrava apenas "Successful Response", sem schema nenhum.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from eolica.application.use_cases import DailyReport
from eolica.domain.health import HealthStatus
from eolica.domain.monitoring import DriftReport, DriftSeverity


class ThresholdInfo(BaseModel):
    """O limiar usado na decisão — exposto para tornar o alerta auditável."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(description="Valor do limiar de erro de reconstrução")
    method: str = Field(description="Como foi derivado: percentile | std_deviations")
    parameter: float = Field(description="O parâmetro do método (ex.: 99.5)")


class HealthSection(BaseModel):
    """Veredito de saúde da turbina."""

    model_config = ConfigDict(frozen=True)

    status: HealthStatus
    reason: str = Field(description="Explicação da decisão, em português")
    exceedances: int = Field(description="Janelas acima do limiar, incluindo picos isolados")
    sustained_anomalies: int = Field(
        description="Janelas em corridas longas o bastante para alarmar"
    )
    evaluated_windows: int
    persistence_window: int = Field(
        description="Janelas consecutivas exigidas para caracterizar alerta"
    )
    previous_period_anomalies: int | None = Field(
        default=None,
        description="Anomalias na véspera. `null` significa desconhecido — nunca zero.",
    )
    previous_period_known: bool
    threshold: ThresholdInfo


class ForecastSection(BaseModel):
    """Previsão de geração para o passo seguinte."""

    model_config = ConfigDict(frozen=True)

    power_kw: float = Field(
        description=(
            "Potência prevista **como deve ser exibida**: nunca negativa. "
            "A regra é aplicada aqui, no servidor — no v1 vivia no prompt do LLM."
        )
    )
    power_kw_measured: float = Field(
        description=(
            "A saída crua do modelo, que pode ser negativa (consumo parasita). "
            "Exposta para que o cliente possa auditar o clamp."
        )
    )
    target_time: datetime
    model_version: str = Field(description="Identidade do modelo que produziu a previsão")


class CoverageSection(BaseModel):
    """Quanto do dia foi efetivamente analisado."""

    model_config = ConfigDict(frozen=True)

    readings: int
    expected_readings: int
    completeness: float = Field(ge=0.0, le=1.0, description="Fração do dia com medição")
    analysed_segments: int = Field(
        description="Trechos contíguos analisados. >1 indica descontinuidade no dia."
    )
    discarded_readings: int = Field(
        description="Leituras em trechos curtos demais para a janela do modelo"
    )
    is_fragmented: bool


class DailyReportResponse(BaseModel):
    """Relatório diário completo."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "day": "2022-01-20",
                "health": {
                    "status": "OK",
                    "reason": "Nenhuma janela acima do limiar.",
                    "exceedances": 0,
                    "sustained_anomalies": 0,
                    "evaluated_windows": 85,
                    "persistence_window": 6,
                    "previous_period_anomalies": 0,
                    "previous_period_known": True,
                    "threshold": {"value": 3.21, "method": "percentile", "parameter": 99.5},
                },
                "forecast": {
                    "power_kw": 0.0,
                    "power_kw_measured": -0.02,
                    "target_time": "2022-01-21T00:00:00Z",
                    "model_version": "persistence-baseline@1",
                },
                "coverage": {
                    "readings": 144,
                    "expected_readings": 144,
                    "completeness": 1.0,
                    "analysed_segments": 1,
                    "discarded_readings": 0,
                    "is_fragmented": False,
                },
                "forecast_unavailable_reason": None,
                "data_range": {"start": "2022-01-14T00:00:00Z", "end": "2022-01-27T23:50:00Z"},
            }
        },
    )

    day: date
    health: HealthSection
    coverage: CoverageSection
    forecast: ForecastSection | None = None
    forecast_unavailable_reason: str | None = Field(
        default=None,
        description=(
            "Motivo da ausência de previsão. O v1 colocava a string 'Indisponível' "
            "dentro do campo numérico, quebrando o tipo."
        ),
    )
    data_range: DataRange

    @classmethod
    def from_domain(cls, report: DailyReport) -> DailyReportResponse:
        """Traduz o objeto de domínio para o contrato público.

        A tradução é explícita e unidirecional: mudar um nome de campo da API
        não vaza para o domínio, e vice-versa.
        """
        verdict = report.health
        return cls(
            day=report.day,
            health=HealthSection(
                status=verdict.status,
                reason=verdict.reason,
                exceedances=verdict.exceedances,
                sustained_anomalies=verdict.sustained_anomalies,
                evaluated_windows=verdict.evaluated_windows,
                persistence_window=verdict.persistence_window,
                previous_period_anomalies=verdict.previous_period_anomalies,
                previous_period_known=verdict.previous_period_known,
                threshold=ThresholdInfo(
                    value=verdict.threshold.value,
                    method=str(verdict.threshold.method),
                    parameter=verdict.threshold.parameter,
                ),
            ),
            coverage=CoverageSection(
                readings=report.coverage.readings,
                expected_readings=report.coverage.expected_readings,
                completeness=report.coverage.completeness,
                analysed_segments=report.coverage.analysed_segments,
                discarded_readings=report.coverage.discarded_readings,
                is_fragmented=report.coverage.is_fragmented,
            ),
            forecast=(
                None
                if report.forecast is None
                else ForecastSection(
                    power_kw=report.forecast.for_display(),
                    power_kw_measured=report.forecast.power.kw,
                    target_time=report.forecast.target_time,
                    model_version=report.forecast.model_version,
                )
            ),
            forecast_unavailable_reason=report.forecast_unavailable_reason,
            data_range=DataRange(start=report.data_range[0], end=report.data_range[1]),
        )


class DataRange(BaseModel):
    """Janela temporal coberta pelo acervo."""

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime


class FeatureDrift(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    score: float
    method: str
    severity: DriftSeverity


class DriftResponse(BaseModel):
    """Comparação entre a distribuição de referência e a recente."""

    model_config = ConfigDict(frozen=True)

    severity: DriftSeverity
    requires_action: bool = Field(
        description="True apenas em drift severo; moderado pede investigação, não retreino"
    )
    worst_feature: str
    features: list[FeatureDrift]

    @classmethod
    def from_domain(cls, report: DriftReport) -> DriftResponse:
        return cls(
            severity=report.severity,
            requires_action=report.requires_action,
            worst_feature=report.worst_feature,
            features=[
                FeatureDrift(
                    feature=name,
                    score=score.value,
                    method=str(score.method),
                    severity=score.severity,
                )
                for name, score in sorted(report.scores.items(), key=lambda item: -item[1].value)
            ],
        )


class ReadinessResponse(BaseModel):
    """Estado de prontidão para receber tráfego."""

    model_config = ConfigDict(frozen=True)

    ready: bool
    checks: dict[str, bool]
    detail: str | None = None


class LivenessResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str = "alive"
    version: str


DailyReportResponse.model_rebuild()
