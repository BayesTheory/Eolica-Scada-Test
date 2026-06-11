"""Plataforma de monitoramento preditivo de turbinas eólicas a partir de dados SCADA.

A arquitetura segue Domain-Driven Design com dependências apontando sempre para
dentro:

    interfaces ──▶ application ──▶ domain ◀── infrastructure

`domain` não importa nada das outras camadas (nem sequer pandas ou torch);
`infrastructure` implementa as portas declaradas por `domain`/`application`.
O teste `tests/architecture/test_layer_dependencies.py` verifica essa regra.
"""

__version__ = "2.0.0"
