"""
Script para engenharia de dados (v2.1 - Corrigido).

Responsável por:
1. Checar se os dados processados já existem para evitar reprocessamento.
2. Carregar dados brutos de alta frequência, tratando o cabeçalho corretamente.
3. Reamostrar os dados para médias de 10 minutos (incluindo todos os status).
4. Salvar o dataset reamostrado completo.
5. Criar e salvar um subconjunto de dados apenas com o status de operação.
"""

import pandas as pd
import os

# --- 1. CONFIGURAÇÃO E METADADOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')

INPUT_CSV_PATH = os.path.join(DATA_DIR, 'Aventa_AV7_IET_OST_SCADA.csv')
# Arquivo 1: Todos os dados reamostrados
OUTPUT_CSV_RESAMPLED = os.path.join(DATA_DIR, 'scada_resampled_10min.csv')
# Arquivo 2: Apenas dados operacionais
OUTPUT_CSV_OPERATIONAL = os.path.join(DATA_DIR, 'scada_operacional_10min.csv')

STATUS_MAP = {
    0: "Initialize system", 1: "Feathered search 1", 2: "Feathered search 2",
    3: "Feathered pos 1", 4: "Function test 1", 5: "Function test 2",
    6: "Feathered pos 2", 7: "Standby pos 1", 8: "Standby pos 2",
    9: "Standby pos 3", 10: "Power Production", 11: "High wind shutdown",
    12: "Shut down", 13: "Alarm – fault condition"
}
STATUS_OPERACAO = 10

# --- 2. FUNÇÕES DO PIPELINE ---

def load_data(path):
    """Carrega os dados e converte a coluna de data."""
    print(f"Carregando dados de: {path}")
    try:
        # CORREÇÃO APLICADA AQUI:
        # Removemos 'header=None' e 'names=...'. Por padrão, o Pandas
        # usará a primeira linha do CSV como cabeçalho, que é o correto.
        df = pd.read_csv(path)
        
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        print("Dados carregados com sucesso.")
        return df
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em {path}.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao carregar os dados: {e}")
        return None

def resample_and_filter(df):
    """Reamostra para 10 minutos e filtra por qualidade/disponibilidade."""
    print("Reamostrando dados para médias de 10 minutos...")
    agregacoes = {
        'WindSpeed': ['mean', 'count'], 'RotorSpeed': 'mean', 'GeneratorSpeed': 'mean',
        'GeneratorTemperature': 'mean', 'PowerOutput': 'mean', 'StatusAnlage': 'mean',
        'PitchDeg': 'mean'
    }
    df_agg = df.resample('10T').agg(agregacoes)
    # Renomeia as colunas para um formato limpo
    df_agg.columns = ['WindSpeed_mean', 'data_points_count', 'RotorSpeed_mean', 'GeneratorSpeed_mean', 'GeneratorTemperature_mean', 'PowerOutput_mean', 'StatusAnlage_mean', 'PitchDeg_mean']
    
    pontos_esperados_10min = 600
    limiar_disponibilidade = 0.9
    df_filtered = df_agg[df_agg['data_points_count'] >= (pontos_esperados_10min * limiar_disponibilidade)].copy()
    print(f"Reamostragem e filtro de qualidade concluídos. Total de amostras: {len(df_filtered)}")
    return df_filtered

def isolate_operation_data(df):
    """Filtra um DataFrame para manter apenas os dados de operação normal."""
    print(f"Isolando dados de operação normal (Status {STATUS_OPERACAO})...")
    df['StatusAnlage_rounded'] = df['StatusAnlage_mean'].round()
    codigos_validos = STATUS_MAP.keys()
    df_cleaned = df[df['StatusAnlage_rounded'].isin(codigos_validos)].copy()
    df_operacao = df_cleaned[df_cleaned['StatusAnlage_rounded'] == STATUS_OPERACAO].copy()
    print(f"Dados de operação isolados: {len(df_operacao)} registros encontrados.")
    return df_operacao

# --- 3. ORQUESTRAÇÃO (BLOCO PRINCIPAL COM LÓGICA DE VERIFICAÇÃO) ---

if __name__ == "__main__":
    print("--- INICIANDO PIPELINE DE PROCESSAMENTO DE DADOS (v2.1) ---")

    # Tarefa 1: Criar o dataset reamostrado completo
    if not os.path.exists(OUTPUT_CSV_RESAMPLED):
        print(f"\nArquivo '{os.path.basename(OUTPUT_CSV_RESAMPLED)}' não encontrado. Gerando agora...")
        raw_df = load_data(INPUT_CSV_PATH)
        if raw_df is not None:
            resampled_df = resample_and_filter(raw_df)
            resampled_df.to_csv(OUTPUT_CSV_RESAMPLED)
            print("Arquivo reamostrado com todos os status foi salvo com sucesso.")
    else:
        print(f"\nArquivo '{os.path.basename(OUTPUT_CSV_RESAMPLED)}' já existe. Pulando etapa de reamostragem.")

    # Tarefa 2: Criar o dataset operacional a partir do reamostrado
    if not os.path.exists(OUTPUT_CSV_OPERATIONAL):
        print(f"\nArquivo '{os.path.basename(OUTPUT_CSV_OPERATIONAL)}' não encontrado. Gerando agora...")
        if os.path.exists(OUTPUT_CSV_RESAMPLED):
            df_resampled_full = pd.read_csv(OUTPUT_CSV_RESAMPLED, index_col='Datetime', parse_dates=True)
            operation_df = isolate_operation_data(df_resampled_full)
            if not operation_df.empty:
                operation_df.to_csv(OUTPUT_CSV_OPERATIONAL)
                print("Arquivo com dados operacionais foi salvo com sucesso.")
        else:
            print("ERRO: Arquivo reamostrado base não existe. Execute o pipeline novamente para criá-lo.")
    else:
        print(f"\nArquivo '{os.path.basename(OUTPUT_CSV_OPERATIONAL)}' já existe. Pulando etapa de isolamento.")

    print("\n--- PIPELINE DE PROCESSAMENTO DE DADOS CONCLUÍDO ---")