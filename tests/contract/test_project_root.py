"""Descoberta da raiz do projeto.

Existe por um bug que passou por 425 testes verdes, pelo lint, pelo mypy e por
todo o CI — e só apareceu quando o container subiu.

`PROJECT_ROOT` era `Path(__file__).resolve().parents[4]`. Num checkout de
código-fonte a contagem acerta. Num pacote instalado, o módulo vive em
`site-packages/eolica/infrastructure/config/settings.py`, e subir quatro níveis
leva a `/opt/venv/lib/python3.11` — um diretório que existe e não tem nada.

O sintoma foi o processo morrer no `lifespan`:

    DataSourceError: Nem o dataset processado nem o sample foram encontrados
    (data_path='/opt/venv/lib/python3.11/data/processed/...')

Nenhum teste local podia pegar: todos rodam a partir do checkout, onde a
aritmética de `parents` acerta por coincidência. Foi preciso instalar o pacote
de verdade — o que só o build da imagem faz.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eolica.infrastructure.config import PROJECT_ROOT
from eolica.infrastructure.config.settings import _MARKERS, _discover_project_root

pytestmark = pytest.mark.contract


class TestNoCheckout:
    def test_encontra_a_raiz_do_repositorio(self) -> None:
        assert (PROJECT_ROOT / "pyproject.toml").is_file()

    def test_os_defaults_de_dados_apontam_para_arquivos_reais(self) -> None:
        """O que quebrou no container: defaults resolvendo para o nada."""
        from eolica.infrastructure.config import Settings

        settings = Settings()
        assert settings.sample_data_path.is_file(), (
            f"o sample default não existe em {settings.sample_data_path}"
        )

    def test_a_raiz_nao_esta_dentro_de_site_packages(self) -> None:
        """A assinatura exata do bug original."""
        assert "site-packages" not in PROJECT_ROOT.parts
        assert PROJECT_ROOT.name != "python3.11"


class TestQuandoNaoHaMarcador:
    """O caso do pacote instalado, onde nenhum ancestral tem os marcadores."""

    def test_cai_no_diretorio_de_trabalho(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Num container, o diretório de trabalho é o `WORKDIR` — onde os dados
        de fato estão. É um fallback útil, não um chute."""
        fake_site_packages = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
        module = fake_site_packages / "eolica" / "infrastructure" / "config" / "settings.py"
        module.parent.mkdir(parents=True)
        module.write_text("", encoding="utf-8")

        workdir = tmp_path / "app"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setattr("eolica.infrastructure.config.settings.__file__", str(module))

        assert _discover_project_root() == workdir

    def test_a_versao_antiga_teria_falhado(self, tmp_path: Path) -> None:
        """Prova que a asserção tem dentes: reproduz a aritmética original e
        mostra que ela produz um caminho sem sentido."""
        module = (
            tmp_path
            / "venv"
            / "lib"
            / "python3.11"
            / "site-packages"
            / "eolica"
            / "infrastructure"
            / "config"
            / "settings.py"
        )
        module.parent.mkdir(parents=True)

        old_behaviour = module.resolve().parents[4]
        assert old_behaviour.name == "python3.11"
        assert not (old_behaviour / "data" / "samples").exists()


class TestMarcadores:
    def test_os_marcadores_existem_na_raiz_real(self) -> None:
        """Se um marcador for renomeado sem atualizar esta lista, a descoberta
        passa a cair no diretório de trabalho silenciosamente."""
        found = [m for m in _MARKERS if (PROJECT_ROOT / m).exists()]
        assert found, f"nenhum marcador de {_MARKERS} existe em {PROJECT_ROOT}"
