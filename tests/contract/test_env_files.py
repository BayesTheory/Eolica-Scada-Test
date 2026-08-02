"""Os arquivos de exemplo de ambiente precisam ser carregáveis.

Este arquivo existe por causa de um bug que passou por uma verificação mal
feita. As variáveis de provisionamento do GCP foram colocadas no `.env`, e a
checagem que "provou" que era seguro usou um caminho POSIX (`/tmp/probe.env`)
que o Python do Windows não resolve. O arquivo nunca foi lido, a checagem passou
vazia, e o `.env` quebrou todo comando que lê configuração — porque `Settings`
usa `extra="forbid"` e qualquer chave desconhecida aborta a inicialização.

A lição não é "teste melhor": é que uma verificação que não pode falhar não
verifica nada. Estes testes usam `tmp_path` do pytest, então o arquivo existe
de fato, e o caso negativo é exercitado explicitamente para provar que a
asserção tem dentes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from eolica.infrastructure.config import PROJECT_ROOT, Settings

pytestmark = pytest.mark.contract

ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
DEPLOY_EXAMPLE = PROJECT_ROOT / ".env.deploy.example"


def _settings_from(path: Path) -> Settings:
    """Constrói `Settings` a partir de um arquivo específico."""

    class Probe(Settings):
        model_config = SettingsConfigDict(
            env_prefix="EOLICA_", env_file=str(path), extra="forbid", frozen=True
        )

    return Probe()


class TestEnvExample:
    def test_o_arquivo_existe(self) -> None:
        assert ENV_EXAMPLE.is_file(), "sem .env.example, ninguém sabe o que configurar"

    def test_settings_carrega_o_exemplo_sem_erro(self) -> None:
        """O caso que o bug quebrou.

        Se alguém adicionar ao `.env.example` uma chave que `Settings` não
        conhece, esta asserção falha aqui — e não na máquina de quem copiou o
        arquivo e tentou rodar a aplicação.
        """
        settings = _settings_from(ENV_EXAMPLE)
        assert settings.environment in {"local", "ci", "staging", "production"}

    def test_o_exemplo_nao_contem_variaveis_de_provisionamento(self) -> None:
        """Elas moram em `.env.deploy` — misturar derruba a aplicação."""
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        for forbidden in ("GCP_PROJECT_ID=", "GCP_PROJECT_NUMBER=", "GITHUB_REPO="):
            assert forbidden not in content, (
                f"'{forbidden}' no .env.example quebra Settings (extra='forbid'). Use .env.deploy."
            )


class TestAAssercaoTemDentes:
    """Prova que o teste acima detectaria o bug original.

    Sem isto, `test_settings_carrega_o_exemplo_sem_erro` poderia passar por
    qualquer motivo — inclusive por não estar lendo arquivo nenhum, que foi
    exatamente o que aconteceu na verificação que falhou.
    """

    def test_chave_desconhecida_no_env_derruba_settings(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.env"
        broken.write_text("EOLICA_LOG_LEVEL=INFO\nGCP_PROJECT_ID=algum-projeto\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="gcp_project_id"):
            _settings_from(broken)

    def test_arquivo_realmente_lido(self, tmp_path: Path) -> None:
        """Confirma que o `env_file` do teste tem efeito.

        É a checagem que faltava: se o caminho não resolvesse, `log_level`
        ficaria no default e o teste passaria sem ler nada.
        """
        env = tmp_path / "ok.env"
        env.write_text("EOLICA_LOG_LEVEL=DEBUG\n", encoding="utf-8")
        assert _settings_from(env).log_level == "DEBUG"


class TestDeployExample:
    def test_o_arquivo_existe(self) -> None:
        assert DEPLOY_EXAMPLE.is_file()

    def test_declara_as_quatro_variaveis_que_o_script_exige(self) -> None:
        """`scripts/setup-gcp.sh` aborta se qualquer uma faltar."""
        content = DEPLOY_EXAMPLE.read_text(encoding="utf-8")
        for required in (
            "GCP_PROJECT_ID=",
            "GCP_PROJECT_NUMBER=",
            "GCP_REGION=",
            "GITHUB_REPO=",
        ):
            assert required in content

    def test_nao_traz_valor_real_de_projeto(self) -> None:
        """O exemplo é versionado num repositório público."""
        content = DEPLOY_EXAMPLE.read_text(encoding="utf-8")
        assert "seu-projeto-id" in content
        assert "fiery-rarity" not in content
