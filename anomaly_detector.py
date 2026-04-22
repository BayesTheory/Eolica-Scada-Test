"""
Anomaly Detector (v2.0) - Orquestrador Integrado

Utiliza dois modelos de ML para uma análise completa:
1.  Modelo de Regressão ('wind-power-forecaster'): Para detectar anomalias de desempenho
    comparando a potência real com a prevista em todo o histórico.
2.  Modelo de Classificação ('wind-fault-predictor'): Para prever a probabilidade
    de uma falha iminente com base nos dados mais recentes.
"""
import pandas as pd
import numpy as np
import mlflow
import pickle
import os
import yaml
from utils import create_sequences

# --- Configurações ---
POWER_MODEL_NAME = "wind-power-forecaster"
FAULT_MODEL_NAME = "wind-fault-predictor"
MODEL_STAGE = "Production"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Limiares de Anomalia
PERFORMANCE_ANOMALY_THRESHOLD_STD = 3.0 # Nº de desvios padrão para erro de potência
FAILURE_PROBABILITY_THRESHOLD = 0.75  # Probabilidade > 75% para gerar alerta de falha

def _load_model_artifacts(model_name, stage):
    """Função auxiliar para carregar um modelo e seus artefatos (scalers, config)."""
    print(f"Carregando artefatos para o modelo '{model_name}'...")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_latest_versions(name=model_name, stages=[stage])[0]
    run_id = model_version.run_id
    
    local_path = client.download_artifacts(run_id, "scalers", ".")
    with open(f"{local_path}/scaler_features.pkl", "rb") as f: scaler_features = pickle.load(f)
    
    scaler_target_path = f"{local_path}/scaler_target.pkl"
    scaler_target = None
    if os.path.exists(scaler_target_path):
        with open(scaler_target_path, "rb") as f: scaler_target = pickle.load(f)

    run_data = client.get_run(run_id).data
    return {
        "model": model, "scaler_features": scaler_features, "scaler_target": scaler_target,
        "run_id": run_id, "params": run_data.params, "tags": run_data.tags
    }

def load_models():
    """Carrega ambos os modelos (regressão e classificação) e seus artefatos."""
    print("Carregando modelos de produção do MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        power_artifacts = _load_model_artifacts(POWER_MODEL_NAME, MODEL_STAGE)
        fault_artifacts = _load_model_artifacts(FAULT_MODEL_NAME, MODEL_STAGE)
        print("Modelos e artefatos carregados com sucesso.")
        return power_artifacts, fault_artifacts
    except Exception as e:
        print(f"\nERRO ao carregar modelos: {e}")
        print("Verifique se ambos os modelos ('wind-power-forecaster' e 'wind-fault-predictor') foram treinados e promovidos para 'Production' na UI do MLflow.")
        return None, None

def check_performance_anomalies(df, artifacts):
    """Usa o modelo de regressão para encontrar desvios de desempenho no histórico."""
    print("Verificando anomalias de desempenho...")
    model, scaler_features, scaler_target = artifacts['model'], artifacts['scaler_features'], artifacts['scaler_target']
    
    with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
    INPUT_WINDOW = config['INPUT_WINDOW_STEPS']
    OUTPUT_HORIZON = config['OUTPUT_HORIZON_STEPS']
    features = config['MODELOS']['3GRU']['features'] # Assume features de um modelo de regressão padrão

    df_scaled = df.copy()
    df_scaled[features] = scaler_features.transform(df_scaled[features])
    X_seq, _ = create_sequences(df_scaled[features].values, INPUT_WINDOW, OUTPUT_HORIZON)
    
    predictions_scaled = model.predict(X_seq)
    predictions_real = scaler_target.inverse_transform(predictions_scaled)
    
    real_values = df['PowerOutput'].values[INPUT_WINDOW:]
    predicted_values = predictions_real[:, 0]
    
    error = real_values - predicted_values
    error_threshold = error.std() * PERFORMANCE_ANOMALY_THRESHOLD_STD
    
    anomaly_indices = np.where(np.abs(error) > error_threshold)[0]
    anomaly_timestamps = df.index[INPUT_WINDOW:][anomaly_indices]
    
    anomalies_df = df.loc[anomaly_timestamps].copy()
    if anomalies_df.empty: return pd.DataFrame()
        
    anomalies_df['AnomalyType'] = 'Performance Deviation'
    anomalies_df['Description'] = [f'Desvio de potência de {error[i]:.2f} kW (limiar: +/-{error_threshold:.2f} kW)' for i in anomaly_indices]
    return anomalies_df[['AnomalyType', 'Description']]

def predict_imminent_failures(df, artifacts):
    """Usa o modelo de classificação para prever a probabilidade de falha futura."""
    print("Prevendo falhas iminentes com o modelo de classificação...")
    model, scaler_features = artifacts['model'], artifacts['scaler_features']
    
    with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
    INPUT_WINDOW = config['INPUT_WINDOW_STEPS']
    features = config['MODELOS']['4.5LSTM_tunado']['features']
    
    recent_data = df.tail(INPUT_WINDOW)
    if len(recent_data) < INPUT_WINDOW:
        print("AVISO: Dados históricos insuficientes para fazer uma previsão de falha.")
        return pd.DataFrame()

    recent_data_scaled = recent_data.copy()
    recent_data_scaled[features] = scaler_features.transform(recent_data_scaled[features])
    input_tensor = np.array([recent_data_scaled[features].values], dtype=np.float32)

    logit = model.predict(input_tensor)[0]
    prob_failure = 1 / (1 + np.exp(-logit))

    print(f"Probabilidade de falha iminente calculada: {prob_failure:.1%}")

    if prob_failure > FAILURE_PROBABILITY_THRESHOLD:
        last_timestamp = df.index[-1]
        anomaly_data = {
            'AnomalyType': 'Imminent Failure Risk',
            'Description': f'ALERTA: Alta probabilidade ({prob_failure:.1%}) de falha nas próximas horas.'
        }
        return pd.DataFrame(anomaly_data, index=[last_timestamp])
        
    return pd.DataFrame()

if __name__ == "__main__":
    power_artifacts, fault_artifacts = load_models()
    
    if power_artifacts and fault_artifacts:
        df_features = pd.read_csv('Data/scada_resampled_10min_features.csv', index_col='Datetime', parse_dates=True)

        print("\n--- INICIANDO DETECÇÃO INTEGRADA DE ANOMALIAS ---")
        
        perf_anomalies = check_performance_anomalies(df_features, power_artifacts)
        imminent_failure_risk = predict_imminent_failures(df_features, fault_artifacts)

        final_report = pd.concat([perf_anomalies, imminent_failure_risk])
        
        if final_report.empty:
            print("\n--- Nenhuma anomalia ou risco iminente detectado ---")
        else:
            final_report.sort_index(inplace=True)
            print("\n--- RELATÓRIO DE ANOMALIAS E RISCOS ---")
            print(final_report)

            output_dir = 'reports'
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, 'anomaly_report.csv')
            final_report.to_csv(report_path)
            print(f"\nRelatório completo salvo em: {report_path}")