"""
Script para engenharia de dados (v2.2 - Features Avançadas e Múltiplos Outputs).

Responsável por:
1. Checar se os dados processados já existem para evitar reprocessamento.
2. Carregar dados brutos de alta frequência.
3. Reamostrar os dados para médias de 10 minutos com filtro de qualidade.
4. Criar features de instabilidade (desvio padrão móvel, etc.) no dataset principal.
5. Salvar o dataset reamostrado completo e enriquecido com features.
6. Criar e salvar um subconjunto de dados contendo apenas o status de operação normal.
"""

import pandas as pd
import os

# --- 1. CONFIGURAÇÃO E METADADOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'Data')

INPUT_CSV_PATH = os.path.join(DATA_DIR, 'Aventa_AV7_IET_OST_SCADA.csv')
# Arquivo 1: Todos os dados reamostrados e com features avançadas
OUTPUT_CSV_RESAMPLED_FEATURES = os.path.join(DATA_DIR, 'scada_resampled_10min_features.csv')
# Arquivo 2: Apenas dados operacionais (derivado do arquivo 1)
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
    # Agrega todas as colunas como média por padrão
    agregacoes = {col: 'mean' for col in df.columns}
    # Especifica que para WindSpeed, também queremos a contagem para o filtro de qualidade
    agregacoes['WindSpeed'] = ['mean', 'count']
    
    df_agg = df.resample('10T').agg(agregacoes)
    
    # Limpa os nomes das colunas multi-indexadas
    df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns.values]
    df_agg.columns = [col.replace('_mean', '') if '_mean' in col else col for col in df_agg.columns]

    pontos_esperados_10min = 600
    limiar_disponibilidade = 0.9
    df_filtered = df_agg[df_agg['WindSpeed_count'] >= (pontos_esperados_10min * limiar_disponibilidade)].copy()
    print(f"Reamostragem e filtro de qualidade concluídos. Total de amostras: {len(df_filtered)}")
    return df_filtered

def create_advanced_features(df):
    """Cria features de instabilidade que podem ser precursoras de falhas."""
    print("Criando features avançadas de instabilidade...")
    window_size_str = '60min' # Janela de 1 hora para calcular a instabilidade
    
    cols_to_feature = ['GeneratorTemperature', 'WindSpeed', 'PowerOutput', 'RotorSpeed', 'PitchDeg']
    
    for col in cols_to_feature:
        if col in df.columns:
            # Desvio Padrão Móvel
            df[f'{col}_std_1h'] = df[col].rolling(window=window_size_str, min_periods=1).std()
            # Taxa de Mudança (derivada em relação a 1 hora atrás)
            df[f'{col}_roc_1h'] = df[col].diff(periods=6)

    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    print("Features avançadas criadas.")
    return df

def isolate_operation_data(df):
    """Filtra um DataFrame para manter apenas os dados de operação normal."""
    print(f"Isolando dados de operação normal (Status {STATUS_OPERACAO})...")
    df['StatusAnlage_rounded'] = df['StatusAnlage'].round()
    
    df_operacao = df[df['StatusAnlage_rounded'] == STATUS_OPERACAO].copy()
    print(f"Dados de operação isolados: {len(df_operacao)} registros encontrados.")
    return df_operacao

# --- 3. ORQUESTRAÇÃO (BLOCO PRINCIPAL COM LÓGICA DE VERIFICAÇÃO) ---

if __name__ == "__main__":
    print("--- INICIANDO PIPELINE DE PROCESSAMENTO DE DADOS (v2.2 - Final) ---")

    # Tarefa 1: Criar o dataset reamostrado completo e com features
    if not os.path.exists(OUTPUT_CSV_RESAMPLED_FEATURES):
        print(f"\nArquivo '{os.path.basename(OUTPUT_CSV_RESAMPLED_FEATURES)}' não encontrado. Gerando agora...")
        raw_df = load_data(INPUT_CSV_PATH)
        if raw_df is not None:
            resampled_df = resample_and_filter(raw_df)
            featured_df = create_advanced_features(resampled_df)
            featured_df.to_csv(OUTPUT_CSV_RESAMPLED_FEATURES)
            print(f"Arquivo reamostrado com features foi salvo com sucesso em '{os.path.basename(OUTPUT_CSV_RESAMPLED_FEATURES)}'.")
    else:
        print(f"\nArquivo '{os.path.basename(OUTPUT_CSV_RESAMPLED_FEATURES)}' já existe. Pulando etapa de reamostragem e feature.")

    # Tarefa 2: Criar o dataset operacional a partir do reamostrado
    if not os.path.exists(OUTPUT_CSV_OPERATIONAL):
        print(f"\nArquivo '{os.path.basename(OUTPUT_CSV_OPERATIONAL)}' não encontrado. Gerando agora...")
        if os.path.exists(OUTPUT_CSV_RESAMPLED_FEATURES):
            # Carrega o arquivo recém-criado ou já existente
            df_featured_full = pd.read_csv(OUTPUT_CSV_RESAMPLED_FEATURES, index_col='Datetime', parse_dates=True)
            operation_df = isolate_operation_data(df_featured_full)
            if not operation_df.empty:
                operation_df.to_csv(OUTPUT_CSV_OPERATIONAL)
                print(f"Arquivo com dados operacionais foi salvo com sucesso em '{os.path.basename(OUTPUT_CSV_OPERATIONAL)}'.")
        else:
            print("ERRO: Arquivo reamostrado base não existe. Execute o pipeline novamente para criá-lo.")
    else:
        print(f"\nArquivo '{os.path.basename(OUTPUT_CSV_OPERATIONAL)}' já existe. Pulando etapa de isolamento.")

    print("\n--- PIPELINE DE PROCESSAMENTO DE DADOS CONCLUÍDO ---")