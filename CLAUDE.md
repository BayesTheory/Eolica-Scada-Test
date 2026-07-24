# Instruções do projeto para o Claude Code

Contexto que não dá para inferir lendo os arquivos. Leia antes de mexer.

## O que é este projeto

Plataforma de monitoramento preditivo de turbinas eólicas a partir de telemetria
SCADA. Detecção de anomalia por reconstrução, previsão de geração e detecção de
drift, servidos por uma API HTTP e uma CLI.

É a v2 de um projeto que existia como scripts soltos na raiz. A v1 está no
histórico do git (até o commit `cf00db8`) e é referenciada nos comentários do
código — quando um comentário diz "no v1...", ele está descrevendo um bug real
que motivou a decisão de design ao lado. **Não remova essas referências**: elas
são o registro de por que o código é assim.

## Arquitetura — a regra que não se quebra

```
interfaces ──▶ application ──▶ domain ◀── infrastructure
```

- **`domain/`** não importa nada além da biblioteca padrão. Nem pandas, nem
  numpy, nem pydantic, nem torch. Se você precisa de uma delas ali, a lógica
  está na camada errada.
- **`application/`** orquestra o domínio através de `Protocol`s. Não conhece
  pandas, MLflow nem HTTP.
- **`infrastructure/`** é onde vivem as bibliotecas pesadas. Implementa portas,
  nunca é importada por domínio ou aplicação.
- **`interfaces/`** é a única camada que pode conhecer todas as outras, e a
  composição acontece exclusivamente em `interfaces/api/container.py`.

Isso é verificado por `tests/architecture/test_layer_dependencies.py`, que lê a
AST de cada módulo. Se você violar, o CI falha com o nome do arquivo. **Não
"conserte" o teste de arquitetura para fazer o código passar** — mova o código.

## Ambiente

- Python 3.11+. O venv do projeto fica em `.venv/`.
- **A variável de ambiente `VIRTUAL_ENV` pode apontar para outro venv.** Se você
  rodar `uv pip install` sem `--python`, ele instala no lugar errado. Sempre:
  ```
  uv pip install --python .venv/Scripts/python.exe -e ".[dev]"   # Windows
  uv pip install --python .venv/bin/python -e ".[dev]"           # Unix
  ```
- Rode testes com `.venv/Scripts/python.exe -m pytest`, não com `pytest` solto.
- `torch`, `xgboost` e `mlflow` estão no extra `[ml]` e **não** são instalados
  por padrão. Isso é deliberado: a suíte inteira roda em ~3s sem eles. Não
  promova essas dependências para o núcleo.

## Dados

- O dataset completo (~65k linhas) vem do Zenodo e é **gitignored**. Rode
  `eolica ingest` para gerar `data/processed/`.
- `data/samples/scada_sample.csv` (1395 linhas, 177 KB) **é versionado** e é o
  que os testes usam. Foi escolhido por conter as três armadilhas do dado real:
  gaps temporais, potência negativa e códigos de status indocumentados.
  Regenere com `python scripts/make_sample.py` se necessário.
- Se `data/processed/` não existir, a aplicação sobe com o sample e registra um
  WARNING. **Nunca torne esse fallback silencioso.**

## Fatos sobre o domínio que não estão óbvios no código

- A turbina é uma **Aventa AV-7** de **6.2 kW**, cut-in 2.0 m/s, cut-out
  12.0 m/s. Fonte: `data/metadata/turbine_metadata.json`.
- **`GeneratorSpeed` não é usada como feature de propósito.** O metadado do
  fabricante marca o canal como `Reliable Measurement = FALSE`. A v1 a usava.
- **Só os códigos de status 10 (produzindo) e 13 (falha) têm semântica
  defensável.** Os códigos 8, 9, 11, 12 e 305 aparecem no dataset e não constam
  de nenhum metadado — por isso `OperatingStatus.UNKNOWN`. O código 9 sozinho é
  38% do dataset. **Não invente significado para eles.**
- **A série tem 30 descontinuidades**, duas de mais de 24h. Toda janela passada
  a um modelo tem que ser contígua — use `ReadingWindow.of()` (que recusa) ou
  `ReadingWindow.split_on_gaps()` (que fatia).
- **Potência negativa é real** (consumo parasita com vento baixo). O dado bruto
  nunca é adulterado; o clamp acontece em `PowerKw.for_display()`.

## Regras de negócio que já moraram no lugar errado

Duas regras viviam **dentro do prompt do LLM** na v1 e foram repatriadas para
`domain/health/services.py`:

1. "Se há anomalia hoje e ontem, o status é EM MANUTENÇÃO."
2. "Nunca mostre potência negativa."

Se aparecer pedido para adicionar uma regra de negócio ao co-piloto, a resposta
é: implemente no domínio e faça o co-piloto ler o resultado.

## Convenções

- Comentários e docstrings em **português**; identificadores em **inglês**.
- Erros: use a hierarquia de `shared/errors.py`. Nunca `raise Exception(...)`,
  nunca `except Exception: pass`. O mapeamento para HTTP acontece só em
  `interfaces/api/errors.py`.
- Respostas de erro seguem RFC 9457 (`application/problem+json`).
- Nada de `print()` — use `structlog` via `infrastructure/observability`.
- Value objects são `@dataclass(frozen=True, slots=True)`.

## Segurança

Este repositório já teve uma chave de API do Google commitada em texto claro
(v1, `co-piloto-llm/co_piloto.py`). A chave foi revogada. O teste
`TestSemSegredoNoCodigo` em `tests/architecture/` varre o fonte procurando
prefixos de segredo conhecidos, e o CI roda `gitleaks` sobre o histórico.

Segredos vêm de variável de ambiente via `SecretStr`, sem default. Se você
precisar de um novo segredo, adicione em `Settings` e em `.env.example` — com o
valor vazio.

## Ao mexer aqui

- **Escreva o teste antes.** O projeto foi construído em TDD e a suíte é o
  contrato.
- Rode `make check` (lint + mypy + testes) antes de commitar.
- Cobertura do domínio tem piso de 95% no CI.
- Commits em Conventional Commits, em português no corpo.
