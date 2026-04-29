"""
Script de Engenharia de Dados (v4.1 - Pipeline Robusto com Saída Base)

Missão: Gerar os datasets fundamentais para os pipelines de Forecasting
e Detecção de Anomalias.

1.  'scada_resampled_10min_base.csv': DADOS COMPLETOS. Contém todos os dados
    reamostrados e colunas de status. Ideal para inferência e análise geral.
2.  'scada_resampled_10min.csv': Dados para a missão de FORECASTING.
    Colunas de status são removidas.
3.  'status_operacional.csv': Dados para a missão de ANOMALY DETECTION.
    Contém APENAS operação normal e colunas de status são removidas.
"""
import pandas as pd
import os
import sys

# --- 1. CONFIGURAÇÃO ---
try:
    # Tenta obter o diretório do script, caso contrário, usa o diretório de trabalho atual
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, 'Data')
os.makedirs(DATA_DIR, exist_ok=True) # Garante que a pasta Data exista

INPUT_CSV_PATH = os.path.join(DATA_DIR, 'Aventa_AV7_IET_OST_SCADA.csv')

# <--- ADICIONADO: Novo arquivo de saída para dados base completos --->
OUTPUT_CSV_BASE = os.path.join(DATA_DIR, 'scada_resampled_10min_base.csv')
# Arquivo de saída para a missão de POWER FORECASTING
OUTPUT_CSV_RESAMPLED = os.path.join(DATA_DIR, 'scada_resampled_10min.csv')
# Arquivo de saída para a missão de FAULT DETECTION
OUTPUT_CSV_OPERATIONAL = os.path.join(DATA_DIR, 'status_operacional.csv')

# Código de status que define "operação normal e saudável"
STATUS_OPERACAO = 10

# --- 2. FUNÇÕES DO PIPELINE ---

def load_data(path):
    """Carrega os dados brutos e os prepara."""
    print(f"Carregando dados brutos de: {path}")
    if not os.path.exists(path):
        print(f"\nERRO CRÍTICO: Arquivo de dados brutos '{os.path.basename(path)}' não encontrado na pasta 'Data'.")
        print("Por favor, certifique-se de que o arquivo SCADA original está na pasta 'Data' antes de continuar.")
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
    
    # Usa '10min' que é a notação moderna do Pandas
    df_agg = df.resample('10min').agg(agregacoes)
    
    # Limpa os nomes das colunas multi-indexadas
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    
    # Filtra amostras que não têm dados suficientes (garante qualidade)
    limiar_qualidade = 0.9 * 600 # 90% dos pontos esperados em 10 min
    df_filtered = df_agg[df_agg['WindSpeed_count'] >= limiar_qualidade].copy()
    
    # Renomeia as colunas para um formato limpo
    df_filtered.columns = [col.replace('_mean', '') for col in df_filtered.columns]
    
    # Cria a coluna de status arredondado que será usada para filtrar
    if 'StatusAnlage' in df_filtered.columns:
        df_filtered['Status_rounded'] = df_filtered['StatusAnlage'].round()
    
    print(f"Reamostragem e filtro de qualidade concluídos. Total de amostras: {len(df_filtered)}")
    return df_filtered

# <--- ADICIONADO: Nova função para salvar o dataset base --->
def generate_base_dataset(df, path):
    """Salva o dataset reamostrado completo, antes de qualquer remoção de colunas."""
    print(f"Gerando dataset base completo para inferência e análise...")
    df.to_csv(path)
    print(f"-> Arquivo '{os.path.basename(path)}' salvo com sucesso.")

def generate_forecasting_dataset(df, path):
    """Salva o dataset geral para a missão de forecasting."""
    print(f"Gerando dataset para Power Forecasting...")
    # Remove colunas de processamento que não são features úteis para forecasting
    cols_to_drop = ['StatusAnlage', 'Status_rounded', 'WindSpeed_count']
    # Usa um novo dataframe para não modificar o original
    df_forecasting = df.drop(columns=cols_to_drop, errors='ignore')
    df_forecasting.to_csv(path)
    print(f"-> Arquivo '{os.path.basename(path)}' salvo com sucesso.")

def generate_anomaly_dataset(df, path):
    """
    Filtra e salva o dataset contendo APENAS dados de operação normal.
    Este é o passo mais crucial do novo pipeline de anomalias.
    """
    print(f"Gerando dataset de operação normal para Anomaly Detection (Status = {STATUS_OPERACAO})...")
    
    # Filtra pelo status e remove colunas de status que se tornaram redundantes
    operation_df = df[df['Status_rounded'] == STATUS_OPERACAO].copy()
    cols_to_drop = ['StatusAnlage', 'Status_rounded', 'WindSpeed_count']
    operation_df = operation_df.drop(columns=cols_to_drop, errors='ignore')
    
    operation_df.to_csv(path)
    print(f"-> Arquivo '{os.path.basename(path)}' salvo com sucesso ({len(operation_df)} registros).")


# --- 3. ORQUESTRAÇÃO ---
if __name__ == "__main__":
    print("--- INICIANDO PIPELINE DE PROCESSAMENTO (v4.1 - Pipeline Robusto com Saída Base) ---")
    
    # Etapa 1: Carregar dados brutos
    raw_df = load_data(INPUT_CSV_PATH)
    
    # Etapa 2: Reamostrar e filtrar (passo comum e caro, feito uma vez)
    resampled_df = resample_and_filter_quality(raw_df)
    
    print("\n--- Gerando arquivos de saída para as missões ---")
    
    # <--- ADICIONADO: Etapa 3 - Salvar o dataset base completo primeiro --->
    generate_base_dataset(resampled_df, OUTPUT_CSV_BASE)

    # Etapa 4: Gerar e salvar o dataset para a missão de Forecasting
    generate_forecasting_dataset(resampled_df, OUTPUT_CSV_RESAMPLED)
    
    # Etapa 5: Gerar e salvar o dataset para a missão de Anomaly Detection
    generate_anomaly_dataset(resampled_df, OUTPUT_CSV_OPERATIONAL)
    
    print("\n--- PIPELINE DE PROCESSAMENTO CONCLUÍDO ---")