"""
Motor de Treino para Detecção de Anomalias (v2.3 - Arquitetura Profissional)

Melhorias v2.3:
- Adicionada classe TimeSeriesDataset para carregamento de dados sob demanda,
  eliminando o risco de estouro de memória com datasets grandes.
- Implementado shuffle=True no DataLoader de treino para melhorar a generalização.
- Suporta múltiplas estratégias (Preditiva e Autoencoder) de forma flexível.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import os
import pickle
import sys
import copy

# ==============================================================================
# CLASSE DE DATASET CUSTOMIZADA (PARA ESCALABILIDADE)
# ==============================================================================
class TimeSeriesDataset(Dataset):
    """
    Dataset customizado para carregar janelas de séries temporais sob demanda.
    Isso evita carregar todo o conjunto de sequências na RAM.
    """
    def __init__(self, data, window_size, mode='autoencoder', cond_indices=None, health_indices=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.mode = mode
        self.cond_indices = cond_indices
        self.health_indices = health_indices
        
        # Calcula o número de amostras possíveis
        if self.mode == 'predictive':
            self.num_samples = len(self.data) - self.window_size
        else: # autoencoder
            self.num_samples = len(self.data) - self.window_size + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == 'predictive':
            # Janela de entrada (X) com features de condição
            x = self.data[idx : idx + self.window_size, self.cond_indices]
            # Alvo (y) é o próximo passo das features de saúde
            y = self.data[idx + self.window_size, self.health_indices]
        else: # autoencoder
            # Janela de entrada (X) e alvo (y) são a mesma janela de dados
            window = self.data[idx : idx + self.window_size]
            x, y = window, window
        return x, y

# ==============================================================================
# MOTOR DE TREINO PRINCIPAL
# ==============================================================================
def train_anomaly_specialist(model_class, model_key, model_config, mission_config, global_config):
    print(f"\n--- Iniciando treino do especialista '{model_key}' para Detecção de Anomalias ---")

    # Lógica de decisão da estratégia
    if 'condition_features' in model_config:
        training_mode = 'predictive'
        print("Estratégia de Treinamento: Preditiva (Causa -> Efeito)")
    elif 'features' in model_config:
        training_mode = 'autoencoder'
        print("Estratégia de Treinamento: Autoencoder (Reconstrução da Assinatura)")
    else:
        print("ERRO: Configuração do modelo inválida.")
        sys.exit(1)

    # Carregamento e preparação dos dados
    DATA_FILE = 'Data/status_operacional.csv'
    df = pd.read_csv(DATA_FILE, index_col='Datetime', parse_dates=True)

    if training_mode == 'predictive':
        cond_features, health_features = model_config['condition_features'], model_config['health_features']
        all_features = cond_features + health_features
    else:
        all_features = model_config['features']

    df = df[all_features]
    
    # Divisão cronológica do DataFrame antes de qualquer outra coisa
    split_index = int(len(df) * 0.8)
    df_train, df_val = df[:split_index], df[split_index:]

    # Scaling: treina no conjunto de treino, aplica em ambos
    scaler = StandardScaler()
    data_train_scaled = scaler.fit_transform(df_train)
    data_val_scaled = scaler.transform(df_val)

    window_size = model_config['input_window_steps']
    
    # Instancia os Datasets customizados
    if training_mode == 'predictive':
        cond_indices = [df.columns.get_loc(c) for c in cond_features]
        health_indices = [df.columns.get_loc(h) for h in health_features]
        train_dataset = TimeSeriesDataset(data_train_scaled, window_size, 'predictive', cond_indices, health_indices)
        val_dataset = TimeSeriesDataset(data_val_scaled, window_size, 'predictive', cond_indices, health_indices)
        input_size, output_size = len(cond_features), len(health_features)
    else: # autoencoder
        train_dataset = TimeSeriesDataset(data_train_scaled, window_size, 'autoencoder')
        val_dataset = TimeSeriesDataset(data_val_scaled, window_size, 'autoencoder')
        input_size, output_size = len(all_features), len(all_features)
    
    print(f"Dados preparados: {len(train_dataset)} amostras de treino, {len(val_dataset)} de validação.")

    # DataLoaders: shuffle=True para o treino, False para a validação
    train_loader = DataLoader(train_dataset, batch_size=model_config['params']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_config['params']['batch_size'], shuffle=False)
    
    # Configuração do modelo, otimizador e scheduler
    model_init_params = model_config['params'].copy()
    model_init_params.update({'input_size': input_size, 'output_size': output_size})
    params_to_remove = ['epochs', 'batch_size', 'learning_rate']
    for param in params_to_remove: model_init_params.pop(param, None)
    model = model_class(**model_init_params)
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config['params']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2)
    
    patience = 10
    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_model_state = None

    # Loop de Treinamento
    with mlflow.start_run(run_name=f"Train_Specialist_{model_key}"):
        mlflow.log_params(model_config['params'])
        mlflow.set_tags(model_config['run_tags'])
        mlflow.set_tag("training_mode", training_mode)
        print("\nIniciando treinamento do modelo especialista...")
        
        for epoch in range(model_config['params']['epochs']):
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = loss_function(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            model.eval()
            val_preds_all, y_val_all = [], []
            with torch.no_grad():
                for X_v_batch, y_v_batch in val_loader:
                    val_predictions = model(X_v_batch)
                    val_preds_all.append(val_predictions)
                    y_val_all.append(y_v_batch)
            
            val_preds_all = torch.cat(val_preds_all, dim=0)
            y_val_all = torch.cat(y_val_all, dim=0)

            val_loss_mse = F.mse_loss(val_preds_all, y_val_all).item()
            val_loss_rmse = np.sqrt(val_loss_mse)
            val_loss_mae = F.l1_loss(val_preds_all, y_val_all).item()
            avg_train_loss = np.mean(train_losses)
            
            print(f"Época [{epoch+1}/{model_config['params']['epochs']}], "
                  f"Train Loss (MSE): {avg_train_loss:.6f}, "
                  f"Val Loss (MSE): {val_loss_mse:.6f}, "
                  f"Val RMSE: {val_loss_rmse:.6f}, "
                  f"Val MAE: {val_loss_mae:.6f}")
            
            mlflow.log_metric("train_loss_mse", avg_train_loss, step=epoch)
            mlflow.log_metric("validation_loss_mse", val_loss_mse, step=epoch)
            mlflow.log_metric("validation_rmse", val_loss_rmse, step=epoch)
            mlflow.log_metric("validation_mae", val_loss_mae, step=epoch)
            
            scheduler.step(val_loss_mse)

            if val_loss_mse < best_val_loss:
                best_val_loss = val_loss_mse
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  -> Novo melhor modelo encontrado! Salvando estado.")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"\nParada antecipada na época {epoch+1}. A perda de validação não melhora há {patience} épocas.")
                break
        
        if best_model_state:
            print(f"\nCarregando o melhor modelo (Val Loss MSE: {best_val_loss:.6f}) para salvamento.")
            model.load_state_dict(best_model_state)

        print("\nTreinamento concluído. Salvando artefatos...")
        scaler_filename = "health_model_scaler.pkl"
        with open(scaler_filename, "wb") as f: pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_filename)
        
        # Pega uma amostra para a assinatura do MLflow
        X_sample, _ = next(iter(DataLoader(train_dataset, batch_size=128)))
        signature = infer_signature(X_sample.numpy(), model(X_sample).detach().numpy())

        mlflow.pytorch.log_model(
            pytorch_model=model, artifact_path="model_specialist", signature=signature,
            input_example=X_sample.numpy()
        )
        
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model_specialist"
        registered_model_name = mission_config['model_name_in_registry']
        mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        print(f"Modelo especialista treinado e registrado com sucesso como '{registered_model_name}'!")