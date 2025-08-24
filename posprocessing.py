"""
Script de Pós-Processamento e Geração de Relatórios.

Este script se conecta ao MLflow para buscar os resultados de um treinamento,
carrega o melhor modelo de um 'run' específico e gera visualizações,
como gráficos de valores reais vs. previstos.
"""
import mlflow
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from utils import create_sequences
import yaml
import torch

# Configure para apontar para o seu servidor MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

def generate_prediction_plot(run_id, config):
    """
    Carrega um modelo de um 'run' do MLflow, recria os dados de teste para
    aquele fold específico e gera um gráfico de Real vs. Previsto.
    """
    print(f"\nGerando gráfico para o Run ID: {run_id}")
    
    # --- 1. Carregar o modelo e os scalers do MLflow ---
    client = mlflow.tracking.MlflowClient()
    
    # Pega a tag 'fold' para saber qual split de dados recriar
    run_data = client.get_run(run_id).data
    fold_tag = int(run_data.tags.get('fold', '1'))
    model_key = run_data.tags.get('model_key', 'unknown')

    # Carrega o modelo PyFunc
    model_uri = f"runs:/{run_id}/model_fold_{fold_tag}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Baixa os artefatos (scalers)
    local_path = client.download_artifacts(run_id, "scalers", ".")
    with open(f"{local_path}/scaler_features.pkl", "rb") as f:
        scaler_features = pickle.load(f)
    with open(f"{local_path}/scaler_target.pkl", "rb") as f:
        scaler_target = pickle.load(f)

    # --- 2. Recriar exatamente o mesmo conjunto de dados de teste ---
    print(f"Recriando o conjunto de teste para o Fold {fold_tag}...")
    
    # Carrega os dados originais
    DATA_FILE_PATH = 'Data/scada_resampled_10min.csv'
    df = pd.read_csv(DATA_FILE_PATH, index_col='Datetime', parse_dates=True)
    
    # Pega as configurações do modelo específico que foi treinado
    model_config = config['MODELOS'][model_key]
    features = model_config['features']
    target = 'PowerOutput_mean'

    # Aplica a mesma escala
    df_scaled = df.copy()
    df_scaled[features] = scaler_features.transform(df[features])
    df_scaled[target] = scaler_target.transform(df[[target]])

    # Cria as mesmas sequências
    X, y = create_sequences(
        df_scaled[features].values,
        config['INPUT_WINDOW_STEPS'],
        config['OUTPUT_HORIZON_STEPS']
    )
    target_index = features.index(target)
    y = y[:, :, target_index]

    # Aplica o mesmo split de validação cruzada para encontrar os índices de teste
    tscv = TimeSeriesSplit(n_splits=config['CV_SPLITS'])
    all_splits = list(tscv.split(X))
    _, test_index = all_splits[fold_tag - 1]
    
    X_test, y_test = X[test_index], y[test_index]
    
    # --- 3. Fazer previsões ---
    print("Fazendo previsões com o modelo carregado...")
    predictions_scaled = model.predict(X_test)
    
    # --- 4. Desnormalizar os dados para plotagem ---
    predictions_real = scaler_target.inverse_transform(predictions_scaled)
    y_test_real = scaler_target.inverse_transform(y_test)

    # Para plotagem, vamos focar no primeiro passo de previsão (T+10 min)
    horizonte_passo_1_pred = predictions_real[:, 0]
    horizonte_passo_1_real = y_test_real[:, 0]
    
    # --- 5. Gerar e Salvar o Gráfico ---
    print("Gerando e salvando o gráfico...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(18, 8))
    
    ax.plot(horizonte_passo_1_real, label='Valores Reais', color='royalblue', linewidth=2)
    ax.plot(horizonte_passo_1_pred, label='Valores Previstos', color='darkorange', linestyle='--', linewidth=2)
    
    rmse = np.sqrt(mean_squared_error(horizonte_passo_1_real, horizonte_passo_1_pred))
    
    ax.set_title(f'Previsão vs. Real (T+10 min) - Modelo: {model_key} - Fold: {fold_tag}\nRMSE: {rmse:.2f} kW', fontsize=16)
    ax.set_xlabel('Amostras de Teste (passos de 10 min)', fontsize=12)
    ax.set_ylabel('Potência (kW)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)

    # Garante que o diretório de saída exista
    output_dir = 'reports'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'report_{model_key}_run_{run_id}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo em: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera relatórios de pós-processamento a partir de um run do MLflow.")
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="O ID do 'run' PAI do MLflow (o que contém os folds) para gerar o relatório."
    )
    args = parser.parse_args()

    # Inicia a conexão com o MLflow
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Carrega a configuração geral do projeto
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Busca os runs filhos (os folds) do run pai fornecido
    child_runs = client.search_runs(
        experiment_ids=client.get_run(args.run_id).info.experiment_id,
        filter_string=f"tags.'mlflow.parentRunId' = '{args.run_id}'",
        order_by=["metrics.rmse_kW ASC"] # Ordena para pegar o melhor fold primeiro
    )

    if not child_runs:
        print(f"ERRO: Nenhum run filho encontrado para o Run ID pai: {args.run_id}")
    else:
        # Pega o melhor fold (primeiro da lista ordenada)
        best_fold_run_id = child_runs[0].info.run_id
        
        print("--- INICIANDO PÓS-PROCESSAMENTO ---")
        print(f"Run Pai ID: {args.run_id}")
        print(f"Analisando o melhor fold (menor RMSE), Run ID: {best_fold_run_id}")
        
        generate_prediction_plot(best_fold_run_id, config)
        
        print("\n--- PÓS-PROCESSAMENTO CONCLUÍDO ---")