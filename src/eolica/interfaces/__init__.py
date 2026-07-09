"""Camada de interface: como o mundo fala com a aplicação.

HTTP e CLI são duas portas de entrada para os *mesmos* casos de uso. Nenhuma
das duas contém regra de negócio — se uma regra existisse só na API, a CLI
teria comportamento diferente, que é a origem do problema do v1: a regra de
"em manutenção" existia apenas no cliente de chat.
"""
