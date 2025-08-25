"""
Ferramenta de Análise de Desempenho de Modelos (v1.0).

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
from sklearn.metrics import mean_squared_error
import yaml
import torch
import sys

# Assume que a função 'create_sequences' existe em 'utils.py'
from utils import create_sequences

# Configura o URI do servidor MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

def generate_prediction_plot(run_id, cv_splits, features, model_key):
    """
    Carrega o melhor modelo de um 'run' do MLflow, recria os dados de teste
    e gera um gráfico de Real vs. Previsto.
    """
    print(f"\nGerando gráfico para o Run ID: {run_id}")
    
    client = mlflow.tracking.MlflowClient()
    
    # Busca os parâmetros e tags do run pai para obter as configurações
    run_data = client.get_run(run_id).data
    input_window_steps = int(run_data.params.get('input_window_steps', 60))
    output_horizon_steps = int(run_data.params.get('output_horizon_steps', 1))
    
    # Carrega o modelo PyFunc do melhor fold
    child_runs = client.search_runs(
        experiment_ids=client.get_run(run_id).info.experiment_id,
        filter_string=f"tags.'mlflow.parentRunId' = '{run_id}'",
        order_by=["metrics.rmse_kW ASC"],
        max_results=1
    )
    if not child_runs:
        print(f"ERRO: Nenhum run filho (fold) encontrado para o Run ID pai: {run_id}")
        return

    best_fold_run = child_runs[0]
    best_fold_run_id = best_fold_run.info.run_id
    fold_tag = best_fold_run.data.tags.get('fold', '1')

    model_uri = f"runs:/{best_fold_run_id}/model_fold_{fold_tag}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Baixa os scalers
    local_path = client.download_artifacts(best_fold_run_id, "scalers", ".")
    with open(f"{local_path}/scaler_features.pkl", "rb") as f:
        scaler_features = pickle.load(f)
    with open(f"{local_path}/scaler_target.pkl", "rb") as f:
        scaler_target = pickle.load(f)

    # Recria o conjunto de dados de teste
    print(f"Recriando o conjunto de teste para o Fold {fold_tag}...")
    DATA_FILE_PATH = 'Data/scada_resampled_10min.csv'
    if not os.path.exists(DATA_FILE_PATH):
        print(f"ERRO: Arquivo de dados '{DATA_FILE_PATH}' não encontrado.")
        return
        
    df = pd.read_csv(DATA_FILE_PATH, index_col='Datetime', parse_dates=True)
    target = 'PowerOutput_mean'

    df_scaled = df.copy()
    df_scaled[features] = scaler_features.transform(df[features])
    df_scaled[target] = scaler_target.transform(df[[target]])

    X, y = create_sequences(
        df_scaled[features].values,
        input_window_steps,
        output_horizon_steps
    )
    target_index = features.index(target)
    y = y[:, :, target_index]

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    all_splits = list(tscv.split(X))
    _, test_index = all_splits[int(fold_tag) - 1]
    
    X_test, y_test = X[test_index], y[test_index]
    
    # Faz as previsões e desnormaliza
    predictions_scaled = model.predict(X_test)
    predictions_real = scaler_target.inverse_transform(predictions_scaled)
    y_test_real = scaler_target.inverse_transform(y_test)

    horizonte_passo_1_pred = predictions_real[:, 0]
    horizonte_passo_1_real = y_test_real[:, 0]
    
    # Gera o gráfico
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
        help="O ID do 'run' PAI do MLflow para gerar o relatório."
    )
    parser.add_argument(
        "--model_key",
        type=str,
        required=True,
        help="A chave do modelo, conforme definido no config.yaml (e.g., '3LSTM')."
    )
    args = parser.parse_args()

    # O script assume que o config.yaml está disponível para pegar as features.
    # É uma dependência menor, mas necessária.
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Pega as features da configuração do modelo
        model_config = config['MODEL_DEFINITIONS'].get(args.model_key)
        if not model_config:
            print(f"ERRO: Chave do modelo '{args.model_key}' não encontrada no config.yaml.")
            sys.exit(1)
        
        # O script original usava 'MODELOS' mas seu novo config usa 'MODEL_DEFINITIONS' e 'MISSIONS'
        # A lógica abaixo busca as features de ambas as seções por robustez.
        features = None
        for mission in config['MISSIONS'].values():
            if args.model_key in mission['model_params']:
                features = mission['model_params'][args.model_key].get('features')
                break
        
        if not features:
            print(f"ERRO: Features não encontradas para o modelo '{args.model_key}' no config.yaml.")
            sys.exit(1)

        print("--- INICIANDO ANÁLISE DE DESEMPENHO ---")
        generate_prediction_plot(args.run_id, config['CV_SPLITS'], features, args.model_key)
        print("\n--- ANÁLISE CONCLUÍDA ---")
        
    except Exception as e:
        print(f"ERRO: Ocorreu um erro na execução do script. Detalhe: {e}")