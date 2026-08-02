"""Os arquivos de que a suíte depende estão de fato versionados — e com a caixa
certa.

Este arquivo nasceu de um bug que passou por 415 testes verdes e só apareceu no
CI.

O diretório `Data/` da v1 foi renomeado para `data/`. No disco, funcionou. Mas
o Windows é case-insensitive: para o git local, `Data/` e `data/` são o mesmo
caminho, então o índice manteve `Data/...`, `git status` não acusou nada, e
tudo passou na máquina de desenvolvimento.

No Linux, o checkout criou `Data/` e o código procurou `data/`. Trinta e seis
testes quebraram com "Sample ausente" — em CI, depois do push.

Duas classes de falha ficam cobertas aqui:

- **arquivo referenciado mas não versionado** — passa localmente, quebra em
  qualquer clone limpo;
- **caixa divergente entre índice e código** — invisível em Windows e macOS,
  fatal em Linux.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Caminhos que a suíte e o modo demo exigem, na caixa exata esperada pelo código.
REQUIRED_TRACKED_FILES = (
    "data/samples/scada_sample.csv",
    "data/metadata/scada_channels.csv",
    "data/metadata/turbine_metadata.json",
    ".env.example",
    ".env.deploy.example",
    "frontend/openapi.json",
    "frontend/src/api/schema.ts",
)


def _tracked_files() -> frozenset[str]:
    """Caminhos no índice do git, como o Linux os veria no checkout."""
    # Caminho absoluto resolvido, e não o literal "git": evita depender da
    # ordem do PATH e transforma "git ausente" num skip explícito em vez de um
    # OSError genérico.
    git = shutil.which("git")
    if git is None:  # pragma: no cover - só em ambiente sem git
        pytest.skip("git não encontrado no PATH")

    try:
        # S603 dispara em toda chamada de subprocess, independente da origem da
        # entrada. Aqui não há entrada externa: o executável vem de
        # `shutil.which` e o argumento é literal.
        result = subprocess.run(  # noqa: S603
            [git, "ls-files"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"git indisponível: {exc}")
    return frozenset(result.stdout.splitlines())


@pytest.fixture(scope="module")
def tracked() -> frozenset[str]:
    return _tracked_files()


@pytest.mark.parametrize("path", REQUIRED_TRACKED_FILES)
def test_arquivo_exigido_esta_versionado(path: str, tracked: frozenset[str]) -> None:
    """Existir no disco não basta: precisa estar no índice, com esta caixa.

    A comparação é exata de propósito. `Data/samples/x.csv` no índice satisfaz
    um teste que ignore maiúsculas, e mesmo assim quebra no Linux.
    """
    assert path in tracked, (
        f"'{path}' não está versionado com esta caixa exata. "
        f"Um clone limpo em Linux não terá esse arquivo. "
        f"Verifique com: git ls-files | grep -i {Path(path).name}"
    )


def test_arquivo_exigido_tambem_existe_no_disco() -> None:
    """Rastreado mas ausente localmente indicaria índice e worktree divergentes."""
    missing = [p for p in REQUIRED_TRACKED_FILES if not (PROJECT_ROOT / p).is_file()]
    assert not missing, f"versionados mas ausentes no disco: {missing}"


def test_nenhum_caminho_difere_apenas_por_caixa(tracked: frozenset[str]) -> None:
    """Dois caminhos que só diferem em maiúsculas colidem em Windows e macOS.

    Um checkout num filesystem case-insensitive receberia os dois arquivos no
    mesmo destino, e um sobrescreveria o outro de forma imprevisível.
    """
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []
    for path in tracked:
        key = path.lower()
        if key in seen and seen[key] != path:
            collisions.append((seen[key], path))
        seen[key] = path

    assert not collisions, f"caminhos colidindo por caixa: {collisions}"


def test_dados_derivados_nao_estao_versionados(tracked: frozenset[str]) -> None:
    """`data/processed/` é saída do pipeline de ingestão, não código-fonte.

    São 25 MB regeneráveis. A v1 versionava 1.378 arquivos de tracking do
    MLflow pela mesma falta de critério.
    """
    leaked = sorted(p for p in tracked if p.startswith("data/processed/") and p.endswith(".csv"))
    assert not leaked, f"dados derivados versionados: {leaked}"
