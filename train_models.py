# Salve este conteúdo como train_models.py

"""
Módulo de Treinamento de Modelos Unificado (v2.3 - Depuração).

Este módulo contém os motores de treinamento para as missões definidas no config.yaml.
A lógica foi refatorada para maior robustez, clareza e flexibilidade, eliminando
nomes de missões "chumbados" no código e melhorando o tratamento de erros.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import os
import pickle
import copy
from tqdm import tqdm
import importlib
import sys
import traceback # <<< Adicionado para depuração detalhada

# Assumimos que 'utils.py' e 'constants.py' existem e estão corretos
from utils import create_sequences
from constants import POWER_FORECASTER_MODEL_NAME

# ==============================================================================
# Classes de Dataset (Auxiliares)
# ==============================================================================

class TimeSeriesDataset(Dataset):
    """
    Dataset customizado para carregar janelas de séries temporais sob demanda,
    evitando estouro de memória.
    """
    def __init__(self, data, window_size, mode='autoencoder', cond_indices=None, health_indices=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.mode = mode
        self.cond_indices = cond_indices
        self.health_indices = health_indices
        
        if self.mode == 'predictive':
            self.num_samples = len(self.data) - self.window_size
        else: # autoencoder
            self.num_samples = len(self.data) - self.window_size + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == 'predictive':
            x = self.data[idx : idx + self.window_size, self.cond_indices]
            y = self.data[idx + self.window_size, self.health_indices]
        else: # autoencoder
            window = self.data[idx : idx + self.window_size]
            x, y = window, window
        return x, y

# ==============================================================================
# Motor de Treinamento: POWER FORECASTING
# ==============================================================================

def _create_lagged_features_for_trees(df_input, features_to_lag, target_col, n_lags):
    """Cria features de lag para modelos baseados em árvore."""
    df_lags = pd.DataFrame(index=df_input.index)
    for col in features_to_lag:
        for lag in range(1, n_lags + 1):
            df_lags[f'{col}_lag_{lag}'] = df_input[col].shift(lag)
    df_lags['target'] = df_input[target_col]
    df_lags.dropna(inplace=True)
    X = df_lags.drop(columns=['target'])
    y = df_lags['target']
    return X, y

def train_forecasting_mission(model_class, model_key, model_params, mission_config, config):
    """Motor de treinamento para a missão de Power Forecasting."""
    print(f"\n🚀 Iniciando missão de Forecasting para o modelo: {model_key}...")
    
    data_path = 'Data/scada_resampled_10min.csv'
    if not os.path.exists(data_path):
        print(f"❌ ERRO CRÍTICO: Arquivo de dados '{data_path}' não encontrado.")
        return

    df = pd.read_csv(data_path, index_col='Datetime', parse_dates=True)
    
    features = model_params.get('features', [])
    target_col_name = model_params.get('target', 'PowerOutput')
    
    if not features or target_col_name not in df.columns:
        print(f"❌ ERRO CRÍTICO: 'features' não definidas ou coluna target '{target_col_name}' não encontrada no dataset.")
        return

    cv_splits = config.get('CV_SPLITS', 4)
    run_tags = model_params.get('run_tags', {})
    model_type = run_tags.get('type', 'undefined')

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    fold_metrics = {'rmse': [], 'mae': [], 'r2': []}

    with mlflow.start_run(run_name=f"{model_key}_CV_run") as parent_run:
        mlflow.log_params(model_params.get('params', {}))
        mlflow.set_tags(run_tags)
        print(f"-> MLflow Run Pai iniciado: {parent_run.info.run_id}")

        if model_type == 'tree_based':
            n_lags = model_params.get('params', {}).get('n_lags', 6)
            X, y = _create_lagged_features_for_trees(df, features, target_col_name, n_lags)
            
            best_model = None
            best_rmse = float('inf')

            for fold, (train_index, test_index) in enumerate(tscv.split(X)):
                print(f"\n--- Iniciando Fold {fold+1}/{cv_splits} ---")
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                model = model_class(**model_params.get('params', {}))
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model

                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                fold_metrics['rmse'].append(rmse); fold_metrics['mae'].append(mae); fold_metrics['r2'].append(r2)
                
                mlflow.log_metric("fold_rmse_kW", rmse, step=fold+1)
                mlflow.log_metric("fold_mae_kW", mae, step=fold+1)
                mlflow.log_metric("fold_r2_score", r2, step=fold+1)
                print(f"Fold {fold+1} Métricas - RMSE: {rmse:.2f} kW, MAE: {mae:.2f} kW, R²: {r2:.3f}")
            
        if best_model:
                # Entregamos o modelo de dentro do nosso wrapper: best_model.model
                mlflow.xgboost.log_model(best_model.model, artifact_path="model")
                model_uri = f"runs:/{parent_run.info.run_id}/model"
                mlflow.register_model(model_uri=model_uri, name=POWER_FORECASTER_MODEL_NAME)
                print(f"\n✅ Melhor modelo '{model_key}' (RMSE: {best_rmse:.2f}) treinado e registrado como '{POWER_FORECASTER_MODEL_NAME}'!")
        else:
            print("AVISO: Lógica de treino para PyTorch ou outros modelos não implementada nesta versão.")

        if fold_metrics['rmse']:
            avg_rmse = np.mean(fold_metrics['rmse'])
            avg_mae = np.mean(fold_metrics['mae'])
            avg_r2 = np.mean(fold_metrics['r2'])
            mlflow.log_metric("avg_rmse_kW", avg_rmse)
            mlflow.log_metric("avg_mae_kW", avg_mae)
            mlflow.log_metric("avg_r2_score", avg_r2)
            print(f"\n--- 📋 Resumo CV para {model_key} ---\nRMSE Médio: {avg_rmse:.2f} kW\nMAE Médio: {avg_mae:.2f} kW\nR² Médio: {avg_r2:.3f}")

# ==============================================================================
# Motor de Treinamento: FAULT DETECTION
# ==============================================================================

def train_fault_detection_mission(model_class, model_key, model_params, mission_config, config):
    """Motor de treinamento para a missão de Fault Detection."""
    print(f"\n🚀 Iniciando missão de Fault Detection para o especialista: {model_key}...")

    data_path = 'Data/status_operacional.csv'
    if not os.path.exists(data_path):
        print(f"❌ ERRO CRÍTICO: Arquivo de dados '{data_path}' não encontrado.")
        return

    df = pd.read_csv(data_path, index_col='Datetime', parse_dates=True)
    
    if 'condition_features' in model_params:
        training_mode = 'predictive'
        features = model_params['condition_features'] + model_params['health_features']
    elif 'features' in model_params:
        training_mode = 'autoencoder'
        features = model_params['features']
    else:
        print("❌ ERRO CRÍTICO: Configuração de features inválida para o modelo de anomalia.")
        return
        
    print(f"-> Estratégia de Treinamento: {training_mode.capitalize()}")
    df = df[features]
    
    split_index = int(len(df) * 0.8)
    df_train, df_val = df.iloc[:split_index], df.iloc[split_index:]

    scaler = StandardScaler()
    data_train_scaled = scaler.fit_transform(df_train)
    data_val_scaled = scaler.transform(df_val)
    
    window_size = model_params.get('input_window_steps', 60)
    batch_size = model_params.get('params', {}).get('batch_size', 32)

    train_dataset = TimeSeriesDataset(data_train_scaled, window_size, mode=training_mode)
    val_dataset = TimeSeriesDataset(data_val_scaled, window_size, mode=training_mode)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"-> Dados preparados: {len(train_dataset)} amostras de treino, {len(val_dataset)} de validação.")

    model_init_params = model_params.get('params', {}).copy()
    model_init_params['input_size'] = model_init_params['output_size'] = len(features)
    train_params_to_remove = ['epochs', 'batch_size', 'learning_rate']
    for param in train_params_to_remove:
        model_init_params.pop(param, None)
    model = model_class(**model_init_params)

    with mlflow.start_run(run_name=f"Train_Specialist_{model_key}"):
        mlflow.log_params(model_params.get('params', {}))
        mlflow.set_tags(model_params.get('run_tags', {}))
        mlflow.set_tag("training_mode", training_mode)
        
        loss_function = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_params.get('params', {}).get('learning_rate', 0.001))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2)
        
        patience = 10
        epochs_no_improve = 0
        best_val_loss = float('inf')
        best_model_state = None
        epochs = model_params.get('params', {}).get('epochs', 20)

        print(f"-> Iniciando treinamento por {epochs} épocas...")
        for epoch in range(epochs):
            model.train()
            train_losses = []
            pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs} Treino")
            for X_batch, y_batch in pbar:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = loss_function(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_v_batch, y_v_batch in val_loader:
                    val_predictions = model(X_v_batch)
                    val_loss = F.mse_loss(val_predictions, y_v_batch).item()
                    val_losses.append(val_loss)
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            print(f"  Época [{epoch+1}/{epochs}] -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            mlflow.log_metric("train_loss_mse", avg_train_loss, step=epoch)
            mlflow.log_metric("validation_loss_mse", avg_val_loss, step=epoch)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"  -- Parada antecipada na época {epoch+1} por falta de melhora --")
                break
        
        if best_model_state:
            print(f"\n-> Carregando o melhor modelo encontrado (Val Loss: {best_val_loss:.6f}) para salvamento.")
            model.load_state_dict(best_model_state)

        scaler_filename = "health_model_scaler.pkl"
        with open(scaler_filename, "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_filename)
        
        X_sample, _ = next(iter(train_loader))
        signature = infer_signature(X_sample.numpy(), model(X_sample).detach().numpy())
        
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model_specialist",
            signature=signature,
            input_example=X_sample.numpy()
        )
      
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model_specialist"
        registered_model_name = mission_config.get('model_name_in_registry')
        if registered_model_name:
            mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            print(f"\n✅ Modelo especialista treinado e registrado com sucesso como '{registered_model_name}'!")
        else:
            print("\n⚠️ AVISO: 'model_name_in_registry' não definido na config. O modelo não foi registrado.")

# ==============================================================================
# ORQUESTRADOR PRINCIPAL DE TREINAMENTO
# ==============================================================================

def run_training_mission(mission_key: str, model_key: str, config: dict):
    """
    Orquestrador principal. Roteia a execução para o motor de treinamento correto
    com base no 'trainer_script' definido no config.yaml, de forma dinâmica.
    """
    mission_config = config.get('MISSIONS', {}).get(mission_key)
    if not mission_config:
        print(f"❌ ERRO: Missão '{mission_key}' não encontrada no config.yaml.")
        return

    model_def = config.get('MODEL_DEFINITIONS', {}).get(model_key)
    if not model_def:
        print(f"❌ ERRO: Definição do modelo '{model_key}' não encontrada no config.yaml.")
        return
        
    model_params = mission_config.get('model_params', {}).get(model_key)
    if not model_params:
        print(f"❌ ERRO: Parâmetros para o modelo '{model_key}' não encontrados na missão '{mission_key}'.")
        return

    mlflow_uri = config.get('MLFLOW_TRACKING_URI')
    experiment_name = mission_config.get('experiment_name')
    if not mlflow_uri or not experiment_name:
        print("❌ ERRO CRÍTICO: 'MLFLOW_TRACKING_URI' ou 'experiment_name' não definidos no config.yaml.")
        return
        
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    print(f"-> MLflow configurado. Experimento: '{experiment_name}'")

    try:
        module = importlib.import_module(model_def['module_name'])
        ModelClass = getattr(module, model_def['class_name'])
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: Não foi possível carregar a classe do modelo '{model_def['class_name']}'. Detalhe: {e}")
        return

    trainer_function_name = mission_config.get('trainer_script')
    if not trainer_function_name:
        print(f"❌ ERRO: 'trainer_script' não definido para a missão '{mission_key}' no config.yaml.")
        return

    try:
        trainer_function = getattr(sys.modules[__name__], trainer_function_name)
        trainer_function(ModelClass, model_key, model_params, mission_config, config)
    except Exception as e:
        print(f"❌ Ocorreu um erro durante a execução da missão. Detalhe: {e}")
        print("--- TRACEBACK COMPLETO ---")
        traceback.print_exc()
        print("--------------------------")