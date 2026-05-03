# Pipeline de Detecção de Anomalias em Turbinas Eólicas (Eolica Scada Test)

Este projeto implementa um pipeline de Machine Learning de ponta a ponta para analisar dados SCADA de turbinas eólicas. O objetivo principal é detectar anomalias operacionais e fornecer alertas preditivos, agindo como um "especialista" em saúde da turbina. A abordagem utiliza um Autoencoder com LSTMs para aprender a "assinatura de operação normal" e identificar desvios com base no erro de reconstrução.

---

## 🚀 Principais Funcionalidades

* **Detecção Preditiva de Anomalias:** O modelo é capaz de identificar sinais de alerta com horas de antecedência antes que uma falha oficial seja registrada no sistema.
* **Pipeline de Dados Robusto:** Um script de pré-processamento (`dataprocessing.py`) limpa os dados brutos e gera datasets otimizados e consistentes para treinamento e inferência.
* **Motor de Treino Profissional:** Utiliza as melhores práticas de MLOps, incluindo um carregador de dados eficiente em memória (`TimeSeriesDataset`), separação cronológica para evitar vazamento de dados, e técnicas de treino avançadas como `Early Stopping` e `AdamW`.
* **Análise de Diagnóstico Completa:** Um script de inferência unificado (`analise_completa.py`) que valida o modelo, gera uma Matriz de Confusão, caça sinais precoces e plota visualizações detalhadas para análise de causa raiz.
* **Versionamento com MLOps:** Integração total com **MLflow** para rastreamento de experimentos e registro de modelos, e **Git/Git LFS** para controle de versão do código e de grandes arquivos.

---

## 📁 Estrutura do Projeto

.
├── Data/
│   ├── Aventa_AV7_IET_OST_SCADA.csv  # Dados brutos (NÃO versionado com Git)
│   ├── scada_resampled_10min_base.csv  # Dados processados para inferência
│   └── status_operacional.csv        # Dados de operação normal para treino
├── mlflow_artifacts_analyzer/
├── mlruns/
├── .gitignore
├── analise_completa.py               # Script principal para inferência e análise
├── config.yaml                       # Arquivo central de configuração
├── dataprocessing.py                 # Script para processar os dados brutos
├── model_autoencoder.py              # Definição da arquitetura do modelo
├── train_anomaly_model.py            # Motor de treinamento do modelo
└── README.md                         # Este arquivo


---

## ⚙️ Instalação e Configuração

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/BayesTheory/Eolica-Scada-Test.git](https://github.com/BayesTheory/Eolica-Scada-Test.git)
    cd Eolica-Scada-Test
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # No Windows
    .\venv\Scripts\activate
    # No macOS/Linux
    # source venv/bin/activate
    ```

3.  **Instale as dependências:**
    Crie um arquivo `requirements.txt` com o conteúdo abaixo e rode o comando `pip install`.

    **`requirements.txt`:**
    ```
    pandas
    numpy
    torch
    scikit-learn
    mlflow
    matplotlib
    seaborn
    pyyaml
    ```

    **Comando de instalação:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Workflow de Uso

O pipeline é executado em 3 etapas principais.

### Passo 1: Processamento de Dados

Execute este script uma única vez para gerar os arquivos CSV necessários a partir dos seus dados brutos.

```bash
python dataprocessing.py
