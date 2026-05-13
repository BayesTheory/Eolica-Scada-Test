import os
import yaml
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
import mlflow
import numpy as np
from datetime import timedelta

# Importa as classes de lógica de negócio
from anomaly_analyzer import AnomalyAnalyzer
from forecaster import Forecaster

# --- CONFIGURAÇÃO E INICIALIZAÇÃO DA API ---

try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("❌ ERRO: Arquivo 'config.yaml' não encontrado.")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERRO ao ler o arquivo 'config.yaml'. Detalhe: {e}")
    sys.exit(1)

MISSIONS = config.get('MISSIONS', {})
ANOMALY_MODEL_NAME = MISSIONS.get('fault_detection', {}).get('model_name_in_registry')
FORECASTING_MODEL_KEY = '2XGboosting'
FORECASTING_MODEL_NAME = 'wind-power-forecaster'
MLFLOW_TRACKING_URI = config.get('MLFLOW_TRACKING_URI', "http://localhost:5000")

DATA_FILE_PATH = 'Data/scada_resampled_10min_base.csv'
try:
    print(f"-> Carregando dataset principal de '{DATA_FILE_PATH}' para a memória...")
    df_principal = pd.read_csv(DATA_FILE_PATH, index_col='Datetime', parse_dates=True)
    print("-> Dataset carregado com sucesso.")
except FileNotFoundError:
    print(f"❌ ERRO FATAL: Arquivo de dados principal não encontrado em '{DATA_FILE_PATH}'.")
    sys.exit(1)

try:
    print("-> Inicializando especialistas (modelos)...")
    anomaly_analyzer = AnomalyAnalyzer(model_name=ANOMALY_MODEL_NAME, mlflow_uri=MLFLOW_TRACKING_URI)
    forecaster = Forecaster(model_name=FORECASTING_MODEL_NAME, mlflow_uri=MLFLOW_TRACKING_URI)

    anomaly_params = MISSIONS.get('fault_detection', {}).get('model_params', {}).get('3LSTM_Autoencoder', {})
    anomaly_features = anomaly_params.get('features')
    anomaly_window_size = anomaly_params.get('input_window_steps')

    forecasting_params = MISSIONS.get('power_forecasting', {}).get('model_params', {}).get(FORECASTING_MODEL_KEY, {})
    forecasting_features = forecasting_params.get('features')
    print("-> Especialistas inicializados com sucesso.")

except Exception as e:
    print(f"❌ ERRO FATAL ao inicializar os especialistas: {e}")
    sys.exit(1)

app = FastAPI(
    title="API dos Especialistas de Turbinas Eólicas",
    description="Serve os resultados dos modelos especialistas em saúde e previsão de turbinas."
)

# --- ENDPOINTS DA API ---

@app.get("/gerar_relatorio_diario/")
def gerar_relatorio_diario_endpoint(data_string: str):
    print(f"\n[API] Recebido pedido para analisar a data: {data_string}")
    
    try:
        data_atual = pd.to_datetime(data_string)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Formato de data inválido: {data_string}")

    df_dia = df_principal.loc[data_string]
    if df_dia.empty:
        return {"erro": f"Nenhum dado encontrado para a data {data_string}."}
    
    # --- MUDANÇA 1: Coletar dados do dia anterior para dar contexto ao LLM ---
    data_anterior_str = (data_atual - timedelta(days=1)).strftime('%Y-%m-%d')
    anomalias_dia_anterior = None
    if data_anterior_str in df_principal.index.strftime('%Y-%m-%d'):
        df_anterior = df_principal.loc[data_anterior_str]
        try:
            erros_anterior = anomaly_analyzer.get_reconstruction_errors(df_anterior, anomaly_features, anomaly_window_size)
            anomalias_dia_anterior = int(np.sum(erros_anterior > anomaly_analyzer.threshold))
        except Exception:
            anomalias_dia_anterior = -1 # Indica erro no dia anterior

    # Análise do dia atual
    try:
        if anomaly_analyzer.threshold is None:
            print("[API] Calculando limiar de anomalia (primeira execução)...")
            df_normal = df_principal[df_principal['Status_rounded'] == 10]
            anomaly_analyzer.calculate_threshold(df_normal, anomaly_features, anomaly_window_size)
        
        erros_dia = anomaly_analyzer.get_reconstruction_errors(df_dia, anomaly_features, anomaly_window_size)
        anomalias_detectadas = np.sum(erros_dia > anomaly_analyzer.threshold)
        
        resumo_anomalia = {
            "status": "ALERTA" if anomalias_detectadas > 0 else "OK",
            "anomalias_detectadas": int(anomalias_detectadas),
            "anomalias_dia_anterior": anomalias_dia_anterior # Novo campo!
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar anomalias para {data_string}: {e}")

    # Lógica de previsão
    try:
        df_historico = df_principal.loc[:data_string]
        n_lags = forecasting_params.get('n_lags', 6)
        previsao = forecaster.predict_next_step(df_historico, forecasting_features, n_lags)
        resumo_previsao = {"previsao_kw": float(previsao)}
    except Exception as e:
        print(f"AVISO: Falha na previsão para {data_string}. Detalhe: {e}")
        resumo_previsao = {"previsao_kw": "Indisponível"}

    relatorio_final = {
        "data_analisada": data_string,
        "periodo_total_dados": {
            "inicio": df_principal.index.min().strftime('%Y-%m-%d'),
            "fim": df_principal.index.max().strftime('%Y-%m-%d')
        },
        "relatorio_saude": resumo_anomalia,
        "relatorio_previsao": resumo_previsao
    }
    print("[API] Relatório gerado e enviado com sucesso.")
    return relatorio_final