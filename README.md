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
