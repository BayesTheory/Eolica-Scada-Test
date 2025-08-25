# 🌬️ Co-piloto Eólico: Pipeline MLOps para Análise de Turbinas

## Sobre o Projeto

O **Co-piloto Eólico** é um sistema de ponta a ponta que utiliza Machine Learning e IA Generativa para monitorar a saúde e prever a performance de turbinas eólicas a partir de dados SCADA.

O projeto implementa um pipeline MLOps completo, desde o processamento dos dados e treinamento de modelos até o deploy via API e a interação através de um assistente de IA conversacional inteligente.

## ✨ Principais Funcionalidades

* **Detecção de Anomalias:** Utiliza um modelo **LSTM Autoencoder** treinado em PyTorch para identificar comportamentos anômalos na operação da turbina, analisando o erro de reconstrução dos dados.
* **Previsão de Geração de Potência:** Emprega um modelo **XGBoost** para prever a geração de energia, permitindo um planejamento mais eficaz da operação e manutenção.
* **Interface de IA Conversacional:** Permite que operadores consultem o status da turbina em linguagem natural através do **"Co-piloto Eólico"**, uma interface construída com a API do Google Gemini.
* **Ciclo de Vida MLOps:** Gerencia todo o ciclo de vida dos modelos com **MLflow**, incluindo o registro de experimentos, artefatos (como scalers) e o versionamento dos modelos.
* **API de Inferência:** Disponibiliza os modelos treinados através de uma **API FastAPI** robusta e otimizada, que serve como o "cérebro" para o Co-piloto.

## 🛠️ Stack Tecnológica

* **Análise e Modelagem:** Python, Pandas, Scikit-learn, PyTorch, XGBoost
* **MLOps:** MLflow
* **API:** FastAPI, Uvicorn
* **IA Generativa:** Google Gemini
* **Orquestração:** Scripts Python para pipelines de dados e treinamento.

## 🚀 Como Rodar o Sistema Completo

Para executar o sistema, você precisará de 3 terminais rodando simultaneamente.

### Pré-requisitos
- Python 3.10+
- Todas as bibliotecas instaladas a partir de um arquivo `requirements.txt` (sugestão).
- Dados brutos (`Aventa_AV7_IET_OST_SCADA.csv`) na pasta `Data/`.

### Etapa 1: Preparar os Dados e Treinar os Modelos

Se esta for a primeira execução, processe os dados e treine os modelos.

```bash
# 1. Processar os dados
python main.py process_data

# 2. Treinar o modelo de detecção de falhas (LSTM Autoencoder)
python main.py train fault_detection 3LSTM_Autoencoder

# 3. Treinar o modelo de previsão de potência (XGBoost)
python main.py train power_forecasting 2XGboosting
```

### Etapa 2: Executar os Serviços em Produção

**Terminal 1 - Servidor de Modelos (MLflow):**
Este terminal serve o banco de dados de modelos que a API irá consultar.
```bash
mlflow ui
```

**Terminal 2 - Servidor da API:**
Este terminal carrega os modelos do MLflow e os expõe através de endpoints.
```bash
uvicorn inference_api:app --reload
```

**Terminal 3 - Co-piloto Eólico (Cliente):**
Este é o terminal com o qual você irá interagir.
```bash
python co_piloto.py
```

Após iniciar, faça perguntas como: `Qual o status da turbina para o dia 2022-02-07?`

## Arquitetura do Sistema

O fluxo de informação ocorre da seguinte maneira:

1.  O **Usuário** faz uma pergunta em linguagem natural no script `co_piloto.py`.
2.  O **Google Gemini** interpreta a pergunta e aciona a ferramenta `gerar_relatorio_diario`.
3.  A ferramenta faz uma requisição HTTP para a **API FastAPI** (`inference_api.py`).
4.  A API utiliza os especialistas (`AnomalyAnalyzer` e `Forecaster`), que carregam os **modelos do MLflow** para processar a solicitação.
5.  A API retorna um relatório estruturado (JSON) para o Co-piloto.
6.  O **Google Gemini** recebe o JSON e o traduz em uma resposta formatada e inteligente para o usuário, seguindo as regras de negócio definidas no prompt.
