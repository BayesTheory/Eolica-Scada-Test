"""
Motor de Treinamento Unificado (v2.8)
- Implementa boas práticas do MLflow: nomes de execução descritivos e tags customizáveis.
- Corrige o vazamento de dados (data leakage) no XGBoost para prevenir overfitting.
- Adiciona métricas MAE e R² e gráfico de convergência da perda.
- Garante que a validação cruzada para todos os modelos use TimeSeriesSplit.
- Adicionada barra de progresso (tqdm).
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import os
import pickle
from tqdm import tqdm
from utils import create_sequences
import matplotlib.pyplot as plt

def get_descriptive_run_name(model_key, config):
    """Cria um nome de execução descritivo com base nos hiperparâmetros."""
    params = config['params']
    model_type = config.get('type', 'pytorch')
    
    if model_type == 'pytorch':
        # Ex: 3GRU_e-10_lr-0.001
        return f"{model_key}_e-{params.get('epochs', 'N/A')}_lr-{params.get('learning_rate', 'N/A')}"
    elif model_type == 'tree_based':
        # Ex: 2XGboosting_est-500_lr-0.05
        return f"{model_key}_est-{params.get('n_estimators', 'N/A')}_lr-{params.get('learning_rate', 'N/A')}"
    elif model_type == 'statistical':
        # Ex: 1Arima_ord-5-1-0
        order_str = "-".join(map(str, params.get('order', '')))
        return f"{model_key}_ord-{order_str}"
    else:
        return f"{model_key}_CV_run"

# ==============================================================================
# MOTOR DE TREINAMENTO PARA PYTORCH (GRU, LSTM, etc.)
# ==============================================================================
def run_pytorch_cv(model_class, model_key, config, input_window_size, output_horizon_size, cv_splits):
    print(f"\nIniciando treinamento com motor PyTorch para o modelo: {model_key}...")
    DATA_FILE_PATH = 'Data/scada_resampled_10min.csv'
    if not os.path.exists(DATA_FILE_PATH):
        raise FileNotFoundError(f"Arquivo de dados '{DATA_FILE_PATH}' não encontrado.")
    
    df = pd.read_csv(DATA_FILE_PATH, index_col='Datetime', parse_dates=True)
    features, target = config['features'], 'PowerOutput_mean'
    
    scaler_features, scaler_target = MinMaxScaler(), MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler_features.fit_transform(df[features])
    df_scaled[target] = scaler_target.fit_transform(df[[target]])

    X, y = create_sequences(df_scaled[features].values, input_window_size, output_horizon_size)
    target_index = features.index(target)
    y = y[:, :, target_index]
    X_tensor, y_tensor = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    fold_metrics = {'rmse': [], 'mae': [], 'r2': []}

    run_name = get_descriptive_run_name(model_key, config)
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params(config['params'])
        mlflow.set_tags(config.get('run_tags', {}))
        mlflow.set_tag("model_key", model_key)
        print(f"MLflow Parent Run ID: {parent_run.info.run_id}")

        for fold, (train_index, test_index) in enumerate(tscv.split(X_tensor)):
            run_name = f"{model_key}_fold_{fold+1}"
            with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                mlflow.set_tags({"model_key": model_key, "fold": fold+1})
                X_train, X_test = X_tensor[train_index], X_tensor[test_index]
                y_train, y_test = y_tensor[train_index], y_tensor[test_index]
                
                model_init_params = config['params'].copy()
                model_init_params['input_size'] = len(features)
                model_init_params['output_size'] = output_horizon_size
                train_params_to_remove = ['epochs', 'batch_size', 'learning_rate']
                for param in train_params_to_remove: model_init_params.pop(param, None)
                model = model_class(**model_init_params)
                
                train_loader = DataLoader(dataset=TensorDataset(X_train, y_train), batch_size=config['params']['batch_size'], shuffle=False)
                loss_function, optimizer = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=config['params']['learning_rate'])
                
                epoch_losses = []
                print(f"\n--- Iniciando Fold {fold+1}/{cv_splits} ---")
                for epoch in tqdm(range(config['params']['epochs']), desc=f"Fold {fold+1} Training"):
                    model.train()
                    batch_losses = []
                    for X_batch, y_batch in train_loader:
                        outputs = model(X_batch); loss = loss_function(outputs, y_batch)
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                        batch_losses.append(loss.item())
                    epoch_losses.append(np.mean(batch_losses))
                
                plt.figure(figsize=(10, 5)); plt.plot(epoch_losses, label='Training Loss')
                plt.title(f'Convergência do Treinamento - Fold {fold+1}'); plt.xlabel('Época'); plt.ylabel('Loss (MSE)')
                plt.legend(); plt.grid(True)
                convergence_plot_path = f"convergence_fold_{fold+1}.png"
                plt.savefig(convergence_plot_path); plt.close()
                mlflow.log_artifact(convergence_plot_path)

                model.eval()
                with torch.no_grad(): predictions_scaled = model(X_test).detach().numpy()
                predictions_real = scaler_target.inverse_transform(predictions_scaled)
                y_test_real = scaler_target.inverse_transform(y_test.numpy())
                
                rmse = np.sqrt(mean_squared_error(y_test_real, predictions_real)); mae = mean_absolute_error(y_test_real, predictions_real); r2 = r2_score(y_test_real, predictions_real)
                fold_metrics['rmse'].append(rmse); fold_metrics['mae'].append(mae); fold_metrics['r2'].append(r2)
                mlflow.log_metric("rmse_kW", rmse); mlflow.log_metric("mae_kW", mae); mlflow.log_metric("r2_score", r2)
                print(f"Fold {fold+1} Metrics - RMSE: {rmse:.2f} kW, MAE: {mae:.2f} kW, R²: {r2:.3f}")
                
                with open("scaler_features.pkl", "wb") as f: pickle.dump(scaler_features, f)
                with open("scaler_target.pkl", "wb") as f: pickle.dump(scaler_target, f)
                mlflow.log_artifact("scaler_features.pkl", "scalers"); mlflow.log_artifact("scaler_target.pkl", "scalers")
                signature = infer_signature(X_test.numpy(), predictions_scaled)
                mlflow.pytorch.log_model(model, artifact_path=f"model_fold_{fold+1}", signature=signature)

        avg_rmse = np.mean(fold_metrics['rmse']); avg_mae = np.mean(fold_metrics['mae']); avg_r2 = np.mean(fold_metrics['r2'])
        mlflow.log_metric("avg_rmse_kW", avg_rmse); mlflow.log_metric("avg_mae_kW", avg_mae); mlflow.log_metric("avg_r2_score", avg_r2)
        print(f"\n--- Resumo CV para {model_key} ---\nRMSE Médio: {avg_rmse:.2f} kW\nMAE Médio: {avg_mae:.2f} kW\nR² Médio: {avg_r2:.3f}")

        best_run_df = mlflow.search_runs(experiment_ids=[parent_run.info.experiment_id], filter_string=f"tags.'mlflow.parentRunId' = '{parent_run.info.run_id}'", order_by=["metrics.rmse_kW ASC"], max_results=1)
        if not best_run_df.empty:
            best_run_id = best_run_df.iloc[0]['run_id']
            mlflow.set_tag("best_run_id", best_run_id)
            fold_tag = mlflow.get_run(best_run_id).data.tags.get('fold', '1')
            model_uri = f"runs:/{best_run_id}/model_fold_{fold_tag}"
            mlflow.register_model(model_uri=model_uri, name="wind-power-forecaster")
            print(f"\nMelhor modelo (Run ID: {best_run_id}) registrado com sucesso!")

# ==============================================================================
# MOTOR DE TREINAMENTO PARA MODELOS ESTATÍSTICOS (ARIMA)
# ==============================================================================
def run_arima_cv(model_class, model_key, config, output_horizon_size, cv_splits):
    print(f"\nIniciando treinamento com motor Estatístico para o modelo: {model_key}...")
    DATA_FILE_PATH = 'Data/scada_resampled_10min.csv'
    df = pd.read_csv(DATA_FILE_PATH, index_col='Datetime', parse_dates=True)
    
    df = df.asfreq('10T')
    df.interpolate(method='linear', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    time_series = df[config['features'][0]]

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    fold_metrics = {'rmse': [], 'mae': [], 'r2': []}

    run_name = get_descriptive_run_name(model_key, config)
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params(config['params'])
        mlflow.set_tags(config.get('run_tags', {}))
        mlflow.set_tag("model_key", model_key)
        print(f"MLflow Parent Run ID: {parent_run.info.run_id}")

        for fold, (train_index, test_index) in enumerate(tscv.split(time_series)):
            with mlflow.start_run(run_name=f"{model_key}_fold_{fold+1}", nested=True):
                print(f"\n--- Iniciando Fold {fold+1}/{cv_splits} ---")
                mlflow.set_tags({"model_key": model_key, "fold": fold+1})
                train_data = time_series.iloc[train_index]; test_data = time_series.iloc[test_index]

                model = model_class(**config['params'])
                model.fit(train_data)
                
                predictions = model.predict(steps=output_horizon_size)
                actuals = test_data.iloc[:output_horizon_size]
                predictions.index = actuals.index
                
                rmse = np.sqrt(mean_squared_error(actuals, predictions)); mae = mean_absolute_error(actuals, predictions); r2 = r2_score(actuals, predictions)
                fold_metrics['rmse'].append(rmse); fold_metrics['mae'].append(mae); fold_metrics['r2'].append(r2)
                mlflow.log_metric("rmse_kW", rmse); mlflow.log_metric("mae_kW", mae); mlflow.log_metric("r2_score", r2)
                print(f"Fold {fold+1} Metrics - RMSE: {rmse:.2f} kW, MAE: {mae:.2f} kW, R²: {r2:.3f}")

        avg_rmse = np.mean(fold_metrics['rmse']); avg_mae = np.mean(fold_metrics['mae']); avg_r2 = np.mean(fold_metrics['r2'])
        mlflow.log_metric("avg_rmse_kW", avg_rmse); mlflow.log_metric("avg_mae_kW", avg_mae); mlflow.log_metric("avg_r2_score", avg_r2)
        print(f"\n--- Resumo CV para {model_key} ---\nRMSE Médio: {avg_rmse:.2f} kW\nMAE Médio: {avg_mae:.2f} kW\nR² Médio: {avg_r2:.3f}")

# ==============================================================================
# MOTOR DE TREINAMENTO PARA MODELOS DE ÁRVORE (XGBOOST) - CORRIGIDO
# ==============================================================================
def _create_lagged_features_corrected(df, features_to_lag, target_col, n_lags):
    """
    Função auxiliar corrigida para criar features de lag SEM VAZAMENTO DE DADOS.
    """
    df_lags = pd.DataFrame(index=df.index)
    for col in features_to_lag:
        for lag in range(1, n_lags + 1):
            df_lags[f'{col}_lag_{lag}'] = df[col].shift(lag)
    df_lags['target'] = df[target_col]
    df_lags.dropna(inplace=True)
    X = df_lags.drop(columns=['target'])
    y = df_lags['target']
    return X, y

def run_xgboost_cv(model_class, model_key, config, cv_splits):
    print(f"\nIniciando treinamento com motor de Árvores para o modelo: {model_key}...")
    DATA_FILE_PATH = 'Data/scada_resampled_10min.csv'
    df = pd.read_csv(DATA_FILE_PATH, index_col='Datetime', parse_dates=True)
    
    X, y = _create_lagged_features_corrected(
        df=df,
        features_to_lag=config['features'],
        target_col='PowerOutput_mean',
        n_lags=6
    )

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    fold_metrics = {'rmse': [], 'mae': [], 'r2': []}

    run_name = get_descriptive_run_name(model_key, config)
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params(config['params'])
        mlflow.set_tags(config.get('run_tags', {}))
        mlflow.set_tag("model_key", model_key)
        print(f"MLflow Parent Run ID: {parent_run.info.run_id}")
        
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            with mlflow.start_run(run_name=f"{model_key}_fold_{fold+1}", nested=True):
                print(f"\n--- Iniciando Fold {fold+1}/{cv_splits} ---")
                mlflow.set_tags({"model_key": model_key, "fold": fold+1})
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model = model_class(**config['params'])
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, predictions)); mae = mean_absolute_error(y_test, predictions); r2 = r2_score(y_test, predictions)
                fold_metrics['rmse'].append(rmse); fold_metrics['mae'].append(mae); fold_metrics['r2'].append(r2)
                mlflow.log_metric("rmse_kW", rmse); mlflow.log_metric("mae_kW", mae); mlflow.log_metric("r2_score", r2)
                print(f"Fold {fold+1} Metrics - RMSE: {rmse:.2f} kW, MAE: {mae:.2f} kW, R²: {r2:.3f}")

        avg_rmse = np.mean(fold_metrics['rmse']); avg_mae = np.mean(fold_metrics['mae']); avg_r2 = np.mean(fold_metrics['r2'])
        mlflow.log_metric("avg_rmse_kW", avg_rmse); mlflow.log_metric("avg_mae_kW", avg_mae); mlflow.log_metric("avg_r2_score", avg_r2)
        print(f"\n--- Resumo CV para {model_key} ---\nRMSE Médio: {avg_rmse:.2f} kW\nMAE Médio: {avg_mae:.2f} kW\nR² Médio: {avg_r2:.3f}")