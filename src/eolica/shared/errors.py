"""Hierarquia de erros do sistema.

A raiz do problema no serviço v1 era que toda falha virava `Exception` genérica
e o handler HTTP devolvia 500 para tudo — inclusive para "essa data não existe
no dataset", que é um 404 de livro. Uma data ausente virava incidente de
servidor, e o prompt do LLM acabou instruído a *narrar* o 500 para o operador.

Aqui cada camada tem sua família de erros e o mapeamento para HTTP acontece
uma única vez, na fronteira (`interfaces.api.errors`). O domínio nunca importa
`fastapi` nem conhece códigos de status.
"""

from __future__ import annotations

from typing import Any


class EolicaError(Exception):
    """Raiz de toda falha esperada do sistema.

    `context` carrega dados estruturados para o log — nunca para a mensagem
    do usuário final, que pode acabar exposta numa resposta HTTP.
    """

    def __init__(self, message: str, /, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({rendered})"


# ─────────────────────────────────────────────────────────────────────────────
# Domínio — invariantes de negócio violadas. Culpa do dado ou da regra, nunca
# da infraestrutura.
# ─────────────────────────────────────────────────────────────────────────────
class DomainError(EolicaError):
    """Uma regra de negócio foi violada."""


class InvalidValueError(DomainError):
    """Um value object recebeu um valor fora do domínio permitido."""


class InsufficientDataError(DomainError):
    """Não há observações suficientes para a operação pedida.

    Distinto de "não há dado nenhum": aqui existe dado, mas menos que a janela
    mínima exigida pelo modelo. É a diferença entre 404 e 422 na fronteira.
    """

    def __init__(self, *, required: int, available: int, subject: str = "observações") -> None:
        super().__init__(
            f"São necessárias no mínimo {required} {subject}, mas só há {available}",
            required=required,
            available=available,
            subject=subject,
        )
        self.required = required
        self.available = available


# ─────────────────────────────────────────────────────────────────────────────
# Aplicação — o caso de uso não pôde ser cumprido, mas nada está quebrado.
# ─────────────────────────────────────────────────────────────────────────────
class ApplicationError(EolicaError):
    """Falha na orquestração de um caso de uso."""


class NotFoundError(ApplicationError):
    """O recurso pedido não existe.

    O bug clássico do v1: `df.loc["2022-02-08"]` levanta `KeyError` quando a
    data não está no índice, e a checagem `if df.empty` logo abaixo jamais
    rodava. Virava 500.
    """

    def __init__(self, resource: str, identifier: object) -> None:
        super().__init__(
            f"{resource} não encontrado para '{identifier}'",
            resource=resource,
            identifier=str(identifier),
        )
        self.resource = resource
        self.identifier = identifier


# ─────────────────────────────────────────────────────────────────────────────
# Infraestrutura — o mundo externo falhou. Retentável, alertável, 5xx.
# ─────────────────────────────────────────────────────────────────────────────
class InfrastructureError(EolicaError):
    """Um recurso externo (registry, disco, rede) falhou."""


class ModelUnavailableError(InfrastructureError):
    """O modelo não pôde ser carregado do registry.

    No v1 isso era silenciado: o `except` do carregador setava `self.model = None`
    e o processo seguia vivo, quebrando só na primeira requisição — com stack
    trace irrelevante, longe da causa. Aqui o serviço falha no readiness probe.
    """

    def __init__(self, model_name: str, reason: str) -> None:
        super().__init__(
            f"Modelo '{model_name}' indisponível: {reason}",
            model_name=model_name,
            reason=reason,
        )
        self.model_name = model_name


class DataSourceError(InfrastructureError):
    """A fonte de dados não pôde ser lida."""


class ContractViolationError(InfrastructureError):
    """Um dado atravessou a fronteira violando o contrato declarado.

    Sempre um bug: ou o produtor do dado mudou sem avisar, ou o contrato está
    errado. Nos dois casos queremos falhar alto e cedo, não propagar `NaN`.
    """

    def __init__(self, contract: str, violations: list[str]) -> None:
        preview = "; ".join(violations[:5])
        suffix = f" (+{len(violations) - 5} outras)" if len(violations) > 5 else ""
        super().__init__(
            f"Contrato '{contract}' violado: {preview}{suffix}",
            contract=contract,
            violation_count=len(violations),
        )
        self.contract = contract
        self.violations = violations


class ConfigurationError(EolicaError, ValueError):
    """A configuração é inválida ou inconsistente.

    Levantado na subida do processo, nunca em runtime: config quebrada deve
    impedir o deploy, não gerar erro na milésima requisição.

    Herda também de `ValueError` porque é exatamente isso — um valor inválido
    passado a um construtor. Assim quem só conhece a hierarquia da stdlib
    (um `except ValueError` num script de treino) continua funcionando.
    """
