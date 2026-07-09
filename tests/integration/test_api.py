"""A API HTTP de ponta a ponta, sobre telemetria real.

Sobe a aplicação inteira — container, calibração, routers, error handlers —
usando o sample versionado. Sem MLflow, sem torch, sem rede.

O v1 não tinha como ter este teste: `import inference_api` abria o CSV,
conectava no MLflow e chamava `sys.exit(1)` em caso de falha, derrubando o
processo do pytest durante a coleta.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eolica.infrastructure.config import Settings
from eolica.interfaces.api.app import create_app

pytestmark = pytest.mark.integration

EXISTING_DAY = "2022-01-20"
DAY_OUTSIDE_ARCHIVE = "2030-06-15"


@pytest.fixture(scope="module")
def client(sample_csv_path: Path) -> Iterator[TestClient]:
    """Aplicação real apontada para o sample versionado."""
    settings = Settings(
        data_path=sample_csv_path,
        sample_data_path=sample_csv_path,
        environment="ci",
        log_format="console",
        health_window_size=12,
        persistence_window=3,
        forecast_n_lags=6,
        drift_reference_days=5,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


class TestProbes:
    def test_liveness_responde_sem_tocar_em_dependencia(self, client: TestClient) -> None:
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_confirma_calibracao(self, client: TestClient) -> None:
        response = client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["checks"]["health_model_calibrated"] is True
        assert body["checks"]["repository_loaded"] is True


class TestRelatorioDiario:
    def test_dia_existente_devolve_relatorio_completo(self, client: TestClient) -> None:
        response = client.get(f"/api/v1/reports/{EXISTING_DAY}")
        assert response.status_code == 200
        body = response.json()
        assert body["day"] == EXISTING_DAY
        assert body["health"]["status"] in {"OK", "ALERTA", "EM_MANUTENCAO"}
        assert body["coverage"]["readings"] > 0

    def test_relatorio_expoe_o_limiar_usado(self, client: TestClient) -> None:
        """Auditabilidade: dá para reproduzir a decisão a partir da resposta."""
        body = client.get(f"/api/v1/reports/{EXISTING_DAY}").json()
        threshold = body["health"]["threshold"]
        assert threshold["method"] == "percentile"
        assert threshold["parameter"] == 99.5
        assert threshold["value"] > 0

    def test_relatorio_reporta_a_janela_de_persistencia_em_vigor(self, client: TestClient) -> None:
        """O parâmetro que o v1 declarava no config e nunca lia."""
        body = client.get(f"/api/v1/reports/{EXISTING_DAY}").json()
        assert body["health"]["persistence_window"] == 3

    def test_previsao_nunca_e_negativa_no_campo_de_exibicao(self, client: TestClient) -> None:
        """A regra que morava no prompt do LLM, agora garantida pelo servidor."""
        body = client.get(f"/api/v1/reports/{EXISTING_DAY}").json()
        if body["forecast"] is not None:
            assert body["forecast"]["power_kw"] >= 0.0

    def test_valor_medido_bruto_continua_acessivel(self, client: TestClient) -> None:
        body = client.get(f"/api/v1/reports/{EXISTING_DAY}").json()
        if body["forecast"] is not None:
            assert "power_kw_measured" in body["forecast"]


class TestCaminhosDeErro:
    def test_dia_fora_do_acervo_e_404_e_nao_500(self, client: TestClient) -> None:
        """O bug central do v1, agora coberto por teste.

        `df.loc["2030-06-15"]` levantava KeyError → 500. O prompt do co-piloto
        chegou a instruir o LLM a explicar esses 500 ao operador como "dados
        corrompidos".
        """
        response = client.get(f"/api/v1/reports/{DAY_OUTSIDE_ARCHIVE}")
        assert response.status_code == 404

    def test_erro_segue_o_formato_problem_details(self, client: TestClient) -> None:
        response = client.get(f"/api/v1/reports/{DAY_OUTSIDE_ARCHIVE}")
        assert response.headers["content-type"].startswith("application/problem+json")
        body = response.json()
        assert body["status"] == 404
        assert body["code"] == "not-found"
        assert body["instance"] == f"/api/v1/reports/{DAY_OUTSIDE_ARCHIVE}"
        assert DAY_OUTSIDE_ARCHIVE in body["detail"]

    def test_data_malformada_e_422_da_validacao_do_framework(self, client: TestClient) -> None:
        """A data é tipada como `date` no path: o FastAPI barra antes do handler.

        No v1 era query string livre repassada direto para `.loc` do pandas.
        """
        assert client.get("/api/v1/reports/nao-e-uma-data").status_code == 422

    def test_data_impossivel_e_rejeitada(self, client: TestClient) -> None:
        assert client.get("/api/v1/reports/2022-02-31").status_code == 422

    def test_erro_interno_nao_vaza_detalhe_de_implementacao(self, client: TestClient) -> None:
        body = client.get(f"/api/v1/reports/{DAY_OUTSIDE_ARCHIVE}").json()
        rendered = str(body)
        assert "Traceback" not in rendered
        assert ".py" not in rendered


class TestDrift:
    def test_endpoint_de_drift_responde(self, client: TestClient) -> None:
        response = client.get("/api/v1/drift")
        assert response.status_code in {200, 422}

    def test_drift_reporta_todas_as_features_quando_ha_historico(self, client: TestClient) -> None:
        response = client.get("/api/v1/drift")
        if response.status_code != 200:
            pytest.skip("sample curto demais para dois períodos de comparação")
        body = response.json()
        assert body["severity"] in {"none", "moderate", "severe"}
        assert len(body["features"]) == 5


class TestContratoOpenAPI:
    def test_openapi_e_gerado_com_schemas(self, client: TestClient) -> None:
        """O v1 gerava `/docs` sem schema nenhum: o handler devolvia um dict."""
        spec = client.get("/openapi.json").json()
        assert "DailyReportResponse" in spec["components"]["schemas"]

    def test_endpoint_documenta_os_codigos_de_erro(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        responses = spec["paths"]["/api/v1/reports/{report_date}"]["get"]["responses"]
        assert {"404", "422", "503"} <= set(responses)


class TestMetricas:
    def test_endpoint_de_metricas_expoe_formato_prometheus(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "eolica_" in response.text
