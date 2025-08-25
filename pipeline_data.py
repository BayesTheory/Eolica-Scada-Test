"""
Módulo de Engenharia de Dados Unificado.
Responsável por carregar dados brutos e gerar os datasets limpos e
preparados para as missões de treinamento.
"""
import pandas as pd
import os
import sys

# --- CONFIGURAÇÃO ---
# Obtém o diretório base para garantir que os caminhos sejam independentes do OS
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, 'Data')
INPUT_CSV_PATH = os.path.join(DATA_DIR, 'Aventa_AV7_IET_OST_SCADA.csv')

# Nomes dos arquivos de saída
OUTPUT_CSV_BASE = os.path.join(DATA_DIR, 'scada_resampled_10min_base.csv')
OUTPUT_CSV_RESAMPLED = os.path.join(DATA_DIR, 'scada_resampled_10min.csv')
OUTPUT_CSV_OPERATIONAL = os.path.join(DATA_DIR, 'status_operacional.csv')

# Código de status que define "operação normal e saudável"
STATUS_OPERACAO = 10

# --- FUNÇÕES DO PIPELINE ---

def load_data(path):
    """Carrega e prepara os dados brutos, garantindo que o arquivo exista."""
    print(f"Carregando dados brutos de: {path}")
    if not os.path.exists(path):
        print(f"\nERRO CRÍTICO: Arquivo de dados '{os.path.basename(path)}' não encontrado na pasta 'Data'.")
        sys.exit(1)
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    return df

def resample_and_filter_quality(df):
    """Reamostra os dados para 10 minutos e filtra por qualidade."""
    print("Reamostrando dados para médias de 10 minutos...")
    agregacoes = {col: 'mean' for col in df.columns if df[col].dtype in ['float64', 'int64']}
    agregacoes['WindSpeed'] = ['mean', 'count']
    
    df_agg = df.resample('10min').agg(agregacoes)
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    
    limiar_qualidade = 0.9 * 600 # 90% dos pontos esperados em 10 min
    df_filtered = df_agg[df_agg['WindSpeed_count'] >= limiar_qualidade].copy()
    df_filtered.columns = [col.replace('_mean', '') for col in df_filtered.columns]
    
    if 'StatusAnlage' in df_filtered.columns:
        df_filtered['Status_rounded'] = df_filtered['StatusAnlage'].round()
    
    print(f"Reamostragem e filtro de qualidade concluídos. Total de amostras: {len(df_filtered)}")
    return df_filtered

def generate_datasets(df_resampled):
    """Gera e salva todos os datasets de saída."""
    print("\n--- Gerando arquivos de saída para as missões ---")
    
    # 1. Dataset base completo
    print(f"Gerando dataset base completo para inferência e análise...")
    df_resampled.to_csv(OUTPUT_CSV_BASE)
    print(f"-> Arquivo '{os.path.basename(OUTPUT_CSV_BASE)}' salvo com sucesso.")
    
    # 2. Dataset para Forecasting
    print(f"Gerando dataset para Power Forecasting...")
    cols_to_drop = ['StatusAnlage', 'Status_rounded', 'WindSpeed_count']
    df_forecasting = df_resampled.drop(columns=cols_to_drop, errors='ignore')
    df_forecasting.to_csv(OUTPUT_CSV_RESAMPLED)
    print(f"-> Arquivo '{os.path.basename(OUTPUT_CSV_RESAMPLED)}' salvo com sucesso.")
    
    # 3. Dataset para Anomaly Detection (apenas dados de operação normal)
    print(f"Gerando dataset de operação normal para Anomaly Detection...")
    operation_df = df_resampled[df_resampled['Status_rounded'] == STATUS_OPERACAO].copy()
    operation_df = operation_df.drop(columns=cols_to_drop, errors='ignore')
    operation_df.to_csv(OUTPUT_CSV_OPERATIONAL)
    print(f"-> Arquivo '{os.path.basename(OUTPUT_CSV_OPERATIONAL)}' salvo com sucesso ({len(operation_df)} registros).")

# --- EXECUÇÃO DO PIPELINE ---
if __name__ == "__main__":
    print("--- INICIANDO PIPELINE DE PROCESSAMENTO DE DADOS ---")
    raw_df = load_data(INPUT_CSV_PATH)
    resampled_df = resample_and_filter_quality(raw_df)
    generate_datasets(resampled_df)
    print("\n--- PIPELINE DE PROCESSAMENTO CONCLUÍDO ---")