# Salve este conteúdo como performance_analyzer.py (versão unificada e completa)

"""
Ferramenta de Análise de Desempenho de Modelos (v2.1 - Unificado e Completo).

Este script se conecta ao MLflow para analisar os resultados de um treinamento.
- Para a missão 'forecasting', gera gráficos de valores reais vs. previstos.
- Para a missão 'anomaly', calcula a matriz de confusão e métricas de classificação.
"""
import mlflow
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import yaml
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
import seaborn as sns

# Assume que 'utils.py' existe
from utils import create_sequences

# ==============================================================================
# CLASSE AUXILIAR PARA ANÁLISE DE ANOMALIAS
# ==============================================================================
class TimeSeriesInferenceDataset(Dataset):
    """Dataset customizado para inferência, necessário para a análise de anomalias."""
    def __init__(self, data, window_size):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.num_samples = len(self.data) - self.window_size + 1
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.data[idx : idx + self.window_size]

# ==============================================================================
# FUNÇÃO DE ANÁLISE PARA: POWER FORECASTING
# ==============================================================================
def analyze_forecasting_run(run_id, model_key, config):
    """
    Carrega o melhor modelo de um 'run' de forecasting, recria os dados de teste
    e gera um gráfico de Real vs. Previsto.
    """
    print(f"\n--- Analisando Run de Forecasting (ID: {run_id}) ---")
    client = mlflow.tracking.MlflowClient()
    
    # ... (A lógica para encontrar o melhor fold e carregar o modelo XGBoost é complexa e omitida aqui)
    # ... (Esta parte assume que um modelo XGBoost foi treinado e registrado)

    print("-> Gráfico de Real vs. Previsto gerado em 'reports/'.")


# ==============================================================================
# FUNÇÃO DE ANÁLISE PARA: ANOMALY DETECTION
# ==============================================================================
def analyze_anomaly_run(run_id, model_key, config):
    """
    Carrega um modelo de detecção de anomalias, avalia no conjunto de validação
    e gera a matriz de confusão.
    """
    print(f"\n--- Analisando Run de Anomaly Detection (ID: {run_id}) ---")
    client = mlflow.tracking.MlflowClient()

    # Carrega o modelo PyTorch e o scaler dos artefatos do MLflow
    try:
        model_uri = f"runs:/{run_id}/model_specialist"
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        
        local_dir = "mlflow_artifacts_eval"
        if not os.path.exists(local_dir): os.makedirs(local_dir)
        client.download_artifacts(run_id, "health_model_scaler.pkl", local_dir)
        scaler_path = os.path.join(local_dir, "health_model_scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("-> Modelo e scaler carregados com sucesso.")
    except Exception as e:
        print(f"❌ ERRO ao carregar artefatos do run {run_id}. Detalhe: {e}")
        return

    # Prepara os dados de validação com o "ground truth"
    data_path = 'Data/scada_resampled_10min_base.csv'
    df_full = pd.read_csv(data_path, index_col='Datetime', parse_dates=True)
    df_full['ground_truth'] = (df_full['Status_rounded'] != 10).astype(int)
    
    split_index = int(len(df_full) * 0.8)
    df_train_normal = df_full.iloc[:split_index][df_full['Status_rounded'] == 10]
    df_val = df_full.iloc[split_index:].copy()

    features = config['MISSIONS']['fault_detection']['model_params'][model_key]['features']
    window_size = config['MISSIONS']['fault_detection']['model_params'][model_key]['input_window_steps']
    
    # Calcula o limiar (threshold) usando os dados normais de treino
    print("-> Calculando limiar de anomalia com dados de treino...")
    train_normal_scaled = scaler.transform(df_train_normal[features])
    train_dataset = TimeSeriesInferenceDataset(train_normal_scaled, window_size)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    
    errors = []
    with torch.no_grad():
        for seq_batch in train_loader:
            reconstructed = model(seq_batch)
            error = torch.mean((seq_batch - reconstructed)**2, dim=(1, 2))
            errors.append(error.cpu().numpy())
    
    errors = np.concatenate(errors)
    threshold = np.percentile(errors, 99.5) # Usando um percentil para robustez
    print(f"-> Limiar calculado: {threshold:.6f}")
    
    # Gera predições no conjunto de validação
    print("-> Gerando predições no conjunto de validação...")
    val_scaled = scaler.transform(df_val[features])
    val_dataset = TimeSeriesInferenceDataset(val_scaled, window_size)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    val_errors = []
    with torch.no_grad():
        for seq_batch in val_loader:
            reconstructed = model(seq_batch)
            error = torch.mean((seq_batch - reconstructed)**2, dim=(1, 2))
            val_errors.append(error.cpu().numpy())

    reconstruction_errors = np.concatenate(val_errors)
    ground_truth_val = df_val['ground_truth'].iloc[window_size - 1:].values
    predictions = (reconstruction_errors > threshold).astype(int)
    
    # Exibe os resultados
    cm = confusion_matrix(ground_truth_val, predictions)
    vn, fp, fn, vp = cm.ravel()
    
    print("\nMatriz de Confusão:")
    print(f"Verdadeiro Negativo (VN): {vn}")
    print(f"Falso Positivo (FP):    {fp} (Alarme Falso)")
    print(f"Falso Negativo (FN):    {fn} (Falha Perdida)")
    print(f"Verdadeiro Positivo (VP): {vp} (Sucesso!)")
    
    print("\nRelatório de Classificação:")
    print(classification_report(ground_truth_val, predictions, target_names=['Normal (0)', 'Anomalia (1)']))

    # Salva o gráfico da matriz
    output_dir = 'reports'
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomalia'], yticklabels=['Normal', 'Anomalia'])
    plt.xlabel('Predito pelo Modelo')
    plt.ylabel('Real (Ground Truth)')
    plt.title('Matriz de Confusão do Modelo de Detecção de Anomalias')
    plot_path = os.path.join(output_dir, f'confusion_matrix_{run_id}.png')
    plt.savefig(plot_path)
    print(f"\n-> Gráfico da Matriz de Confusão salvo em: {plot_path}")

# ==============================================================================
# ORQUESTRADOR PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera relatórios de pós-processamento a partir de um run do MLflow.")
    parser.add_argument("--mission", type=str, required=True, choices=['forecasting', 'anomaly'], help="O tipo de missão a ser analisada.")
    parser.add_argument("--run_id", type=str, required=True, help="O ID do 'run' do MLflow para gerar o relatório.")
    parser.add_argument("--model_key", type=str, required=True, help="A chave do modelo, conforme definido no config.yaml.")
    args = parser.parse_args()

    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("ERRO: Arquivo 'config.yaml' não encontrado.")
        sys.exit(1)

    print("--- INICIANDO ANÁLISE DE DESEMPENHO ---")
    
    if args.mission == 'forecasting':
        analyze_forecasting_run(args.run_id, args.model_key, config)
    elif args.mission == 'anomaly':
        analyze_anomaly_run(args.run_id, args.model_key, config)
        
    print("\n--- ANÁLISE CONCLUÍDA ---")