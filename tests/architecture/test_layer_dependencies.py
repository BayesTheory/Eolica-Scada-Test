"""As regras de dependência da arquitetura, verificadas por teste.

Uma arquitetura em camadas que só existe no README é uma convenção — e
convenções erodem no primeiro `from ..infrastructure import` escrito às onze da
noite. Estes testes leem a AST de cada módulo e falham o CI quando uma seta de
dependência aponta para o lado errado.

A regra é uma só, e vale para tudo:

    interfaces ──▶ application ──▶ domain ◀── infrastructure

`domain` não conhece ninguém. `application` conhece `domain`. `infrastructure`
implementa portas de `domain`/`application` mas não é importada por elas.
`interfaces` pode conhecer todos — é lá que a composição acontece.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "eolica"

# Bibliotecas de terceiros que o domínio jamais pode importar. Não é lista
# exaustiva: o teste usa "não é stdlib e não é eolica" como critério.
STDLIB = set(sys.stdlib_module_names)


def _modules_under(package: str) -> list[Path]:
    return sorted((SOURCE_ROOT / package).rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Módulos raiz importados por um arquivo, ignorando blocos TYPE_CHECKING.

    Imports sob `if TYPE_CHECKING:` não existem em runtime e não criam
    acoplamento real — só servem para anotação de tipo.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            for child in node.body:
                for descendant in ast.walk(child):
                    descendant.__dict__["_type_checking_only"] = True

    roots: set[str] = set()
    for node in ast.walk(tree):
        if node.__dict__.get("_type_checking_only"):
            continue
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _eolica_subpackages(path: Path) -> set[str]:
    """Subpacotes de `eolica` importados em runtime por um arquivo."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            for child in node.body:
                for descendant in ast.walk(child):
                    descendant.__dict__["_type_checking_only"] = True

    packages: set[str] = set()
    for node in ast.walk(tree):
        if node.__dict__.get("_type_checking_only"):
            continue
        module = None
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("eolica."):
                    packages.add(alias.name.split(".")[1])
            continue
        if module and module.startswith("eolica."):
            packages.add(module.split(".")[1])
    return packages


class TestDominioEIndependente:
    """O domínio é o núcleo: não depende de nada além da stdlib."""

    @pytest.mark.parametrize("module", _modules_under("domain"), ids=lambda p: p.name)
    def test_dominio_nao_importa_biblioteca_de_terceiros(self, module: Path) -> None:
        """Nem pandas, nem numpy, nem torch, nem pydantic.

        É o que mantém a suíte de domínio em milissegundos e o que permite ler
        a regra de negócio sem saber que existe um LSTM do outro lado.
        """
        foreign = {
            root for root in _imported_roots(module) if root not in STDLIB and root != "eolica"
        }
        assert not foreign, (
            f"{module.relative_to(SOURCE_ROOT)} importa terceiros: {sorted(foreign)}. "
            "O domínio só pode usar a biblioteca padrão."
        )

    @pytest.mark.parametrize("module", _modules_under("domain"), ids=lambda p: p.name)
    def test_dominio_so_importa_dominio_e_shared(self, module: Path) -> None:
        allowed = {"domain", "shared"}
        violations = _eolica_subpackages(module) - allowed
        assert not violations, (
            f"{module.relative_to(SOURCE_ROOT)} importa {sorted(violations)}. "
            "O domínio não pode conhecer application, infrastructure nem interfaces."
        )


class TestAplicacaoNaoConheceInfraestrutura:
    @pytest.mark.parametrize("module", _modules_under("application"), ids=lambda p: p.name)
    def test_aplicacao_nao_importa_infra_nem_interfaces(self, module: Path) -> None:
        """Casos de uso falam com portas, nunca com adaptadores.

        É o que permite testar `GenerateDailyReport` inteiro com fakes em
        memória — 17 testes que rodam em meio segundo sem CSV nem MLflow.
        """
        forbidden = {"infrastructure", "interfaces"}
        violations = _eolica_subpackages(module) & forbidden
        assert not violations, (
            f"{module.relative_to(SOURCE_ROOT)} importa {sorted(violations)}. "
            "Injete pela porta em vez de importar o adaptador."
        )

    @pytest.mark.parametrize("module", _modules_under("application"), ids=lambda p: p.name)
    def test_aplicacao_nao_importa_pandas_em_runtime(self, module: Path) -> None:
        """DataFrame é detalhe de persistência; caso de uso fala em entidades."""
        assert "pandas" not in _imported_roots(module)


class TestInfraestruturaNaoConheceInterfaces:
    @pytest.mark.parametrize("module", _modules_under("infrastructure"), ids=lambda p: p.name)
    def test_infra_nao_importa_interfaces(self, module: Path) -> None:
        violations = _eolica_subpackages(module) & {"interfaces"}
        assert not violations, (
            f"{module.relative_to(SOURCE_ROOT)} importa interfaces — a seta está invertida."
        )


class TestComposicaoAcontecemUmLugarSo:
    def test_apenas_o_composition_root_amarra_infra_a_casos_de_uso(self) -> None:
        """Só `interfaces/api/container.py` (e a CLI) conhecem os dois lados.

        Se este teste começar a falhar apontando outro arquivo, a fiação está
        vazando para fora do composition root.
        """
        wiring: list[str] = []
        for module in _modules_under("interfaces"):
            packages = _eolica_subpackages(module)
            if {"infrastructure", "application"} <= packages:
                wiring.append(module.relative_to(SOURCE_ROOT).as_posix())

        assert set(wiring) <= {
            "interfaces/api/container.py",
            "interfaces/cli/main.py",
        }, f"composição vazou para: {sorted(wiring)}"


class TestSemSegredoNoCodigo:
    """Regressão do incidente que originou este refactor."""

    @pytest.mark.parametrize("module", sorted(SOURCE_ROOT.rglob("*.py")), ids=lambda p: p.name)
    def test_nenhum_arquivo_contem_chave_de_api_literal(self, module: Path) -> None:
        """O v1 tinha `os.environ['GOOGLE_API_KEY'] = "AIzaSy..."` no fonte,
        commitado num repositório público.

        Chaves do Google começam com `AIza`; tokens da OpenAI com `sk-`.
        """
        content = module.read_text(encoding="utf-8")
        for marker in ("AIzaSy", "sk-proj-", "ghp_", "AKIA"):
            assert marker not in content, (
                f"{module.relative_to(SOURCE_ROOT)} parece conter um segredo literal "
                f"('{marker}'). Segredos vêm de variável de ambiente."
            )
