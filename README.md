# Eólica SCADA

Monitoramento preditivo de turbinas eólicas a partir de telemetria SCADA:
detecção de anomalia por reconstrução, previsão de geração e detecção de drift —
servidos por uma API HTTP e uma CLI, com arquitetura em camadas verificada por
teste.

[![CI](https://github.com/BayesTheory/Eolica-Scada-Test/actions/workflows/ci.yml/badge.svg)](https://github.com/BayesTheory/Eolica-Scada-Test/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-165%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## O problema

Uma turbina eólica gera telemetria continuamente — vento, potência, rotação,
temperatura do gerador, ângulo de pitch. Duas perguntas importam ao operador:

1. **Esta máquina está se comportando como quando estava saudável?**
   Um autoencoder aprende a "assinatura" da operação normal; quando o erro de
   reconstrução sobe de forma sustentada, algo mudou.
2. **Quanta energia ela vai gerar no próximo intervalo?**
   Necessário para planejar despacho e janela de manutenção.

E uma terceira, que só aparece depois que o sistema está em produção:

3. **O modelo ainda descreve o mundo que está vendo?**
   Um modelo treinado em 2022 continua respondendo com toda a confiança em 2024.
   Sem medir drift, ninguém descobre que a distribuição mudou embaixo dele.

## Arquitetura

Domain-Driven Design com dependências apontando sempre para dentro.

```mermaid
flowchart LR
    subgraph interfaces["interfaces"]
        API["FastAPI<br/>routers · DI · Problem Details"]
        CLI["Typer<br/>report · drift · ingest"]
    end

    subgraph application["application"]
        UC["Casos de uso<br/>GenerateDailyReport · CheckDrift"]
        P1["Ports<br/>ScadaRepository · Clock · Metrics"]
    end

    subgraph domain["domain — Python puro, zero terceiros"]
        T["turbine<br/>ReadingWindow · PowerKw · Status"]
        H["health<br/>AnomalyThreshold · evaluate_health"]
        F["forecasting<br/>PowerForecast · Horizon"]
        M["monitoring<br/>PSI · Kolmogorov-Smirnov"]
    end

    subgraph infrastructure["infrastructure"]
        REPO["CsvScadaRepository"]
        FS["Feature store<br/>LagFeatureView"]
        ML["Modelos<br/>baseline · torch · xgboost"]
        OBS["structlog · Prometheus"]
    end

    API --> UC
    CLI --> UC
    UC --> P1
    UC --> domain
    REPO -.implementa.-> P1
    ML -.implementa.-> H
    ML -.implementa.-> F
    FS --> ML

    style domain fill:#1a4d2e,stroke:#4ade80,color:#fff
    style application fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style infrastructure fill:#4a3520,stroke:#fbbf24,color:#fff
    style interfaces fill:#3d2645,stroke:#c084fc,color:#fff
```

A regra é **verificada por teste**, não prometida no README:
[`tests/architecture/test_layer_dependencies.py`](tests/architecture/test_layer_dependencies.py)
lê a AST de cada módulo e falha o CI se uma seta apontar para o lado errado.

> `domain/` não importa pandas, numpy, torch nem pydantic. Só a biblioteca
> padrão. É o que mantém a suíte de regras de negócio em **0,6 segundo** e o que
> permite trocar torch por ONNX sem tocar numa linha de lógica.

## Começando

```bash
git clone https://github.com/BayesTheory/Eolica-Scada-Test.git
cd Eolica-Scada-Test

make setup                      # venv + dependências
make test                       # 165 testes, ~3s
make report DAY=2022-01-20      # relatório no terminal
make serve                      # API em http://localhost:8000/docs
```

Funciona logo após o clone: o repositório versiona um recorte real de duas
semanas de telemetria (`data/samples/`), e a aplicação sobe com ele quando o
dataset completo não está presente — registrando um `WARNING` explícito.

Para o dataset inteiro:

```bash
# baixe de https://zenodo.org/records/15700928 para data/raw/
eolica ingest
```

Com Docker:

```bash
docker compose up -d            # API + MLflow + Prometheus
curl localhost:8000/api/v1/reports/2022-01-20 | jq
```

## O que ele responde

```console
$ eolica report 2022-01-20

2022-01-20  —  EM_MANUTENCAO
  Anomalia sustentada por 21 janela(s) hoje e 11 no período anterior:
  indica intervenção em curso.

  janelas avaliadas ....... 35
  acima do limiar ......... 21
  anomalias sustentadas ... 21
  limiar .................. 8.188941
  véspera ................. 11 anomalia(s)

  cobertura ............... 43.8% (63/144 leituras, 2 segmento(s))
  previsão ................ 0.000 kW @ 2022-01-20 23:00 [moving-average-6@1]
```

Repare em **cobertura 43,8% e 2 segmentos**: aquele dia tem um buraco de horas
no meio. O relatório diz isso em vez de fingir que analisou um dia inteiro.

### API

| Método | Rota | O que faz |
|---|---|---|
| `GET` | `/api/v1/reports/{date}` | Relatório diário: saúde, cobertura, previsão |
| `GET` | `/api/v1/drift` | PSI por feature, referência × período recente |
| `GET` | `/health/live` | Liveness — não toca modelo nem disco |
| `GET` | `/health/ready` | Readiness — só 200 depois de calibrar o detector |
| `GET` | `/metrics` | Exposição Prometheus |
| `GET` | `/docs` | OpenAPI com schemas e exemplos |

Erros seguem [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457):

```json
{
  "type": ".../docs/errors.md#not-found",
  "title": "Recurso não encontrado",
  "status": 404,
  "detail": "Telemetria não encontrado para '2030-06-15'",
  "instance": "/api/v1/reports/2030-06-15",
  "code": "not-found",
  "resource": "Telemetria",
  "identifier": "2030-06-15"
}
```

## Decisões de engenharia

| Decisão | Por quê |
|---|---|
| **Domínio sem dependências de terceiros** | Regra de negócio legível por quem entende de turbina, não de tensores. Suíte de domínio em 0,6s. |
| **Feature store com assinatura versionada** | Uma única implementação de lag. Um teste prova que treino e serving produzem o mesmo vetor, bit a bit. |
| **`ReadingWindow` recusa janela furada** | A série tem 30 descontinuidades. Uma janela que atravessa um buraco produz erro de reconstrução alto — indistinguível de falha real. |
| **Limiar calibrado no `lifespan`** | Calcular na primeira requisição faz o usuário pagar a varredura de 32 mil registros como latência, e cria estado mutável compartilhado. |
| **Janela de persistência** | Num sinal de 10 em 10 minutos, um pico isolado é ruído de sensor. Seis janelas consecutivas = uma hora de desvio real. |
| **Baseline explícito (z-score, persistência)** | Sem régua, "MSE 0.003" não significa nada. E permite a API subir e ser testada sem GPU. |
| **`torch`/`mlflow` em extra opcional** | O CI roda a suíte inteira em segundos. Quem quer treinar instala `[ml]`. |
| **Erros tipados por camada** | Dia inexistente é 404, dia fragmentado é 422, registry fora é 503. Nenhum é 500. |
| **Segredos só por `SecretStr`** | Sem default, sem `repr`, sem chegar perto do fonte. |

## Testes

```console
$ make test
165 passed in 2.86s
```

| Suíte | O que cobre | Tempo |
|---|---|---|
| `tests/unit/domain` | Regras de negócio puras | 0,6s |
| `tests/unit/featurestore` | Ausência de skew treino/serving | 1,2s |
| `tests/unit/application` | Casos de uso com fakes em memória | 0,5s |
| `tests/contract` | Contrato de dado contra telemetria **real** | 1,2s |
| `tests/integration` | API completa via `TestClient` | 3,2s |
| `tests/architecture` | Setas de dependência + ausência de segredos | 0,7s |

Os testes de contrato rodam sobre dado de verdade, escolhido justamente por
conter o que dado sintético bem-comportado não tem: gaps, potência negativa e
códigos de status indocumentados.

## De onde isto veio

A v1 deste repositório era um pipeline funcional — LSTM Autoencoder, XGBoost,
MLflow, FastAPI, co-piloto com Gemini — construído como scripts na raiz. Este
reescrita nasceu de uma auditoria dele. Alguns achados, porque explicam
decisões que de outra forma pareceriam excesso de zelo:

- **`python main.py process_data` não rodava.** Chamava `pipeline_data.main()`,
  função que não existia no módulo. O comando estava documentado no README.
- **Data inexistente virava HTTP 500.** `df.loc["2022-02-08"]` levanta
  `KeyError`; a checagem `if df_dia.empty` logo abaixo nunca era alcançada.
- **O prompt do LLM instruía o modelo a narrar esse 500 ao operador** como
  "os dados para esse dia estão corrompidos". Não estavam — o dia não existia.
- **Duas regras de negócio moravam no prompt do LLM**: o critério de "em
  manutenção" e a proibição de exibir potência negativa. Valiam só no chat, e
  eram invisíveis para qualquer outro consumidor da API.
- **`persistence_window: 6` estava no `config.yaml` e nenhuma linha o lia.**
- **Features de lag eram construídas em dois lugares diferentes**, com `n_lags`
  vindo de caminhos distintos da config. Coincidiam em 6 por acidente — ambas
  caíam no mesmo default porque a chave não existia no arquivo.
- **`GeneratorSpeed` era usada como feature** apesar de o metadado do fabricante
  marcá-la como `Reliable Measurement = FALSE`.
- **1.378 arquivos de tracking do MLflow versionados** no git, sem `.gitignore`.
- **Uma chave de API do Google em texto claro**, commitada num repositório
  público. Foi revogada; o CI agora roda `gitleaks` no histórico.

Cada um desses aparece como comentário no ponto do código que o corrige, e a
maioria virou um teste com nome próprio. A v1 continua no histórico do git.

## Estrutura

```
src/eolica/
├── domain/              # regras de negócio — só stdlib
│   ├── turbine/         # ReadingWindow, PowerKw, OperatingStatus
│   ├── health/          # threshold, persistência, veredito + port
│   ├── forecasting/     # previsão, horizonte + port
│   └── monitoring/      # PSI, Kolmogorov-Smirnov
├── application/         # casos de uso + ports de infra
├── infrastructure/      # adaptadores: CSV, feature store, modelos, observabilidade
└── interfaces/          # FastAPI + CLI (composition root)

tests/
├── unit/ contract/ integration/ architecture/
docker/                  # Dockerfile multi-stage, Prometheus
docs/adr/                # decisões de arquitetura registradas
scripts/                 # geração do sample versionado
```

## Dataset

[Aventa AV-7 — IET-OST Research Wind Turbine](https://zenodo.org/records/15700928)
(Zenodo). Turbina de 6.2 kW, rotor de 12,9 m, hub a 18 m. Telemetria a 1 Hz
reamostrada para médias de 10 minutos, de dezembro/2021 a julho/2023.

Canais nomeados segundo **IEC 61400-25** — o mapeamento de nomes internos para o
padrão está em [`data/metadata/scada_channels.csv`](data/metadata/scada_channels.csv)
e é honrado pelo contrato de dado.

## Licença

MIT.
