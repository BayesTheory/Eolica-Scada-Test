"""Camada de aplicação: casos de uso.

Cada classe aqui responde a uma pergunta do negócio ("qual o estado da turbina
no dia X?") orquestrando o domínio e as portas. Não contém regra de negócio —
regra fica no domínio — e não conhece HTTP, CSV nem MLflow.
"""
