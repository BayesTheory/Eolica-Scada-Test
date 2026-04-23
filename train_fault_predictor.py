"""
Motor de Treinamento para Previsão de Falhas (v1.2 - Foco na Transição)

Responsável por:
1.  Definir o alvo como a "transição para o estado de falha", focando apenas
    no momento que precede o início da falha.
2.  Lidar com o desbalanceamento de classes usando ponderação (class weights).
3.  Treinar e avaliar modelos de classificação (ex: LSTMTunado) usando TimeSeriesSplit.
4.  Salvar artefatos (modelo e scalers) e registrar a melhor versão.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from utils import create_sequences
import os
import pickle

def get_descriptive_run_name(model_key, config):
    params = config['params']
    return f"{model_key}_e-{params.get('epochs', 'N/A')}_lr-{params.get('learning_rate', 'N/A')}"

def train_fault_predictor(model_class, model_key, config, input_window_size, cv_splits):
    print(f"\n--- Iniciando Treinamento do Previsor de Falhas: {model_key} ---")
    
    # --- 1. Carregamento e Preparação dos Dados (LÓGICA DO ALVO REFINADA) ---
    DATA_FILE = 'Data/scada_resampled_10min_features.csv'
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Arquivo de dados '{DATA_FILE}' não encontrado. Execute 'data_processing.py' primeiro.")
        
    df = pd.read_csv(DATA_FILE, index_col='Datetime', parse_dates=True)
    
    STATUS_FAULT = 13
    HORIZON_STEPS = 6 # Olhar 1 hora (6 * 10 min) para o futuro

    df['Status_rounded'] = df['StatusAnlage'].round()
    df['is_fault'] = (df['Status_rounded'] == STATUS_FAULT).astype(int)
    
    # Identifica a MUDANÇA para o estado de falha (0 -> 1)
    df['fault_start'] = (df['is_fault'].diff() == 1).astype(int)
    
    # O alvo é 1 se UM INÍCIO de falha ocorrer no horizonte futuro
    df['target'] = df['fault_start'].rolling(window=HORIZON_STEPS, min_periods=1).max().shift(-HORIZON_STEPS)
    
    # Só queremos treinar com exemplos que eram NORMAIS
    df = df[df['is_fault'] == 0]
    
    df.dropna(subset=['target'], inplace=True)
    df['target'] = df['target'].astype(int)

    features = config['features']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    X_seq, y_seq_ignored = create_sequences(df[features].values, input_window_size, 1)
    y_seq = df['target'].values[input_window_size:]
    
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

    # --- 2. Lidar com Desbalanceamento de Classes ---
    neg_count = np.sum(y_seq == 0)
    pos_count = np.sum(y_seq == 1)
    
    if pos_count < 10: # Se tivermos muito poucos eventos de falha
        raise ValueError(f"Foram encontrados apenas {pos_count} eventos de início de falha no dataset. É muito pouco para treinar um modelo robusto.")
        
    pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32)
    print(f"Dataset focado em transições: {neg_count} casos normais, {pos_count} casos de pré-falha. Peso da classe de falha: {pos_weight:.2f}")

    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # ... O resto do script (a partir da seção 3. Treinamento com TimeSeriesSplit)
    # permanece exatamente o mesmo da versão anterior.
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    run_name = get_descriptive_run_name(model_key, config)
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params(config['params'])
        mlflow.set_tags(config.get('run_tags', {}))
        mlflow.set_tag("model_type", "FaultClassifier")

        for fold, (train_index, test_index) in enumerate(tscv.split(X_tensor)):
            with mlflow.start_run(run_name=f"{model_key}_fold_{fold+1}", nested=True):
                X_train, X_test = X_tensor[train_index], X_tensor[test_index]
                y_train, y_test = y_tensor[train_index], y_tensor[test_index]
                
                model_params = config['params'].copy()
                train_params_to_remove = ['epochs', 'batch_size', 'learning_rate']
                for param in train_params_to_remove: model_params.pop(param, None)
                model_params['input_size'] = len(features)
                model_params['output_size'] = 1
                
                model = model_class(**model_params)
                optimizer = torch.optim.Adam(model.parameters(), lr=config['params']['learning_rate'])
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=config['params']['batch_size'], shuffle=False)

                print(f"\n--- Iniciando Fold {fold+1}/{cv_splits} ---")
                for epoch in tqdm(range(config['params']['epochs']), desc=f"Fold {fold+1} Training"):
                    model.train()
                    for X_batch, y_batch in train_loader:
                        optimizer.zero_grad()
                        logits = model(X_batch)
                        loss = loss_function(logits, y_batch)
                        loss.backward()
                        optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    logits = model(X_test)
                    probs = torch.sigmoid(logits).squeeze()
                    preds = (probs > 0.5).long()
                
                y_test_numpy = y_test.squeeze().numpy()
                auc = roc_auc_score(y_test_numpy, probs.cpu().numpy()) if np.any(y_test_numpy) else 0.5
                f1 = f1_score(y_test_numpy, preds.cpu().numpy(), zero_division=0)
                precision = precision_score(y_test_numpy, preds.cpu().numpy(), zero_division=0)
                recall = recall_score(y_test_numpy, preds.cpu().numpy(), zero_division=0)
                
                mlflow.log_metrics({"auc": auc, "f1_score": f1, "precision": precision, "recall": recall})
                print(f"Fold {fold+1} - AUC: {auc:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

                with open("scaler_features.pkl", "wb") as f: pickle.dump(scaler, f)
                mlflow.log_artifact("scaler_features.pkl", artifact_path="scalers")
                mlflow.pytorch.log_model(model, "model")

        print("\nTreinamento de todos os folds concluído. Registrando o melhor modelo...")
        best_run_df = mlflow.search_runs(
            experiment_ids=[parent_run.info.experiment_id],
            filter_string=f"tags.'mlflow.parentRunId' = '{parent_run.info.run_id}'",
            order_by=["metrics.f1_score DESC", "metrics.auc DESC"],
            max_results=1
        )
        if not best_run_df.empty:
            best_run_id = best_run_df.iloc[0]['run_id']
            model_uri = f"runs:/{best_run_id}/model"
            mlflow.register_model(model_uri=model_uri, name="wind-fault-predictor")
            print(f"Melhor modelo (Run ID: {best_run_id}) registrado com sucesso como 'wind-fault-predictor'.")
        else:
            print("AVISO: Não foi possível encontrar o melhor run para registrar.")