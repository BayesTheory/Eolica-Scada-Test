"""Camada de infraestrutura: adaptadores para o mundo externo.

Aqui — e só aqui — vivem pandas, torch, xgboost, mlflow, disco e rede. Cada
módulo implementa uma porta declarada por `domain` ou `application`, e nenhum é
importado por elas: a composição acontece em `interfaces`.
"""
