# ADR 0001 — O domínio não importa bibliotecas de terceiros

**Status:** aceito · **Data:** 2026-08-01

## Contexto

Na v1 não havia camada de domínio. A regra que decidia se uma turbina estava
saudável estava distribuída entre `inference_api.py` (que comparava arrays numpy
dentro de um handler HTTP), `anomaly_analyzer.py` (que misturava carregamento de
modelo, escalonamento e cálculo de erro) e o prompt de sistema do co-piloto
(que continha o critério de "em manutenção").

Testar qualquer uma dessas regras exigia levantar FastAPI, conectar ao MLflow e
carregar um modelo PyTorch. Na prática, ninguém testou.

## Decisão

`src/eolica/domain/` importa **apenas a biblioteca padrão** e `eolica.shared`.

Isso inclui recusar numpy e pandas. Percentil, desvio padrão populacional, PSI e
a estatística de Kolmogorov–Smirnov são reimplementados em Python puro — cerca
de 90 linhas no total, testadas contra valores conhecidos e contra o
comportamento documentado do numpy.

A conversão DataFrame → entidades acontece na fronteira, em
`infrastructure/persistence/csv_repository.py`.

## Consequências

**A favor**

- A suíte de regras de negócio roda em 0,6 s. Isso muda como se trabalha: dá
  para rodar a cada `Ctrl+S`.
- A regra de negócio fica legível para quem entende de turbina eólica e não de
  tensores.
- Trocar PyTorch por ONNX, ou CSV por TimescaleDB, não toca uma linha do
  domínio.
- O teste de arquitetura consegue enunciar a regra de forma binária: "nenhum
  import fora da stdlib". Regras vagas não são verificáveis.

**Contra**

- Há duplicação deliberada: `_percentile` existe aqui e no numpy. Mitigado por
  um teste que fixa a compatibilidade numérica — se divergir, quebra.
- Converter DataFrame em objetos custa alocação. Para 65 mil linhas é
  irrelevante; para milhões, seria preciso reavaliar (provavelmente com um
  adaptador que opere em batch sem materializar entidades).
- Exige disciplina. Por isso a regra é testada, não documentada.

## Alternativas consideradas

- **Permitir numpy no domínio.** Rejeitada: numpy puxa uma ABI compilada, torna
  a regra menos legível (`np.percentile(errors, 99.5)` esconde o método de
  interpolação, que importa) e enfraquece a regra de arquitetura para algo
  discricionário.
- **Domínio anêmico com lógica nos serviços de aplicação.** Rejeitada: é o que
  a v1 tinha de fato, e foi como o critério de manutenção acabou num prompt.
