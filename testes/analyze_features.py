"""
Script de Análise e Seleção de Features (v2.0 - Acadêmico e Text-Only)

Este script realiza uma análise rigorosa e quantitativa para identificar as features
mais preditivas para o alvo de 'pré-falha', sem gerar plots.

MÉTODOS UTILIZADOS:
1.  Filtro (ANOVA F-test): Mede a diferença de médias entre classes.
2.  Filtro (Mutual Information): Captura qualquer relação (linear ou não-linear).
3.  Embarcado (XGBoost Importance): Extrai a importância de features de um modelo complexo.
4.  Empacotamento (Recursive Feature Elimination - RFE): Cria um ranking definitivo
    de features através de eliminação iterativa.

COMO USAR:
1.  Certifique-se de que os dados foram processados com 'data_processing.py'.
2.  Execute o script no terminal: python analyze_features.py
3.  Analise a TABELA DE RESUMO FINAL. Ela consolida todos os resultados e
    fornece um ranking claro para a tomada de decisão.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import os
import sys
import warnings

# Suprime warnings de pacotes para uma saída mais limpa
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. CONFIGURAÇÃO ---
DATA_FILE = 'Data/scada_resampled_10min_features.csv'
TARGET_COL = 'pre_fault_target'
STATUS_FAULT = 13

# Lista de todas as features candidatas que queremos analisar
FEATURES_TO_ANALYZE = [
    'PowerOutput', 'WindSpeed', 'RotorSpeed', 'GeneratorTemperature', 'PitchDeg',
    'GeneratorTemperature_std_1h', 'WindSpeed_std_1h', 'PowerOutput_std_1h',
    'RotorSpeed_std_1h', 'PitchDeg_std_1h',
    'GeneratorTemperature_roc_1h', 'WindSpeed_roc_1h', 'PowerOutput_roc_1h',
    'RotorSpeed_roc_1h', 'PitchDeg_roc_1h'
]

if __name__ == "__main__":
    print("--- INICIANDO ANÁLISE DE FEATURES (MODO ACADÊMICO / TEXT-ONLY) ---")

    # --- 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
    if not os.path.exists(DATA_FILE):
        print(f"\nERRO: Arquivo de dados '{DATA_FILE}' não encontrado. Execute 'python data_processing.py' primeiro.")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE, index_col='Datetime', parse_dates=True)

    if any(col not in df.columns for col in FEATURES_TO_ANALYZE + [TARGET_COL, 'Status_rounded']):
        print(f"\nERRO: O arquivo de dados está incompleto. Faltam colunas.")
        print("Delete os arquivos .csv da pasta 'Data' e rode 'python data_processing.py'.")
        sys.exit(1)
        
    df_analysis = df[df['Status_rounded'] != STATUS_FAULT].copy()
    df_analysis.dropna(inplace=True)

    X = df_analysis[FEATURES_TO_ANALYZE]
    y = df_analysis[TARGET_COL]

    # Escala as features para os testes que são sensíveis à escala
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print(f"\nAnalisando {len(X)} amostras com {int(y.sum())} casos de pré-falha.")

    # --- 3. EXECUÇÃO DOS MÉTODOS DE ANÁLISE ---
    
    # Método 1: Testes Estatísticos de Filtro
    print("\nExecutando testes estatísticos univariados...")
    f_scores, _ = f_classif(X_scaled, y)
    mi_scores = mutual_info_classif(X_scaled, y)
    
    # Método 2: Importância por XGBoost
    print("Calculando importância de features com XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        n_estimators=100,
        max_depth=4
    )
    xgb_model.fit(X, y)
    xgb_importances = xgb_model.feature_importances_
    
    # Método 3: Recursive Feature Elimination (RFE)
    print("Executando Recursive Feature Elimination (RFE)...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    rfe = RFE(estimator=lr_model, n_features_to_select=1, step=1)
    rfe.fit(X_scaled, y)
    rfe_ranks = rfe.ranking_

    # --- 4. SÍNTESE E RESULTADO FINAL ---
    
    # Cria o DataFrame de resumo final
    results_df = pd.DataFrame(index=X.columns)
    results_df['ANOVA_F_score'] = f_scores
    results_df['Mutual_Information'] = mi_scores
    results_df['XGBoost_Importance'] = xgb_importances
    results_df['RFE_Rank'] = rfe_ranks
    
    # Ordena o DataFrame pelo ranking do RFE, que é o mais definitivo
    results_df.sort_values(by='RFE_Rank', ascending=True, inplace=True)
    
    # Formata a saída para melhor legibilidade
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 10)

    print("\n\n" + "="*50)
    print("      TABELA DE RESUMO E RANKING DE FEATURES")
    print("="*50)
    print("Features ordenadas da mais importante (topo) para a menos importante (base).\n")
    print("Interpretação:")
    print("- RFE_Rank: O ranking definitivo (1 = melhor).")
    print("- XGBoost_Importance, Mutual_Information, ANOVA_F_score: Quanto maior, melhor.")
    print("-" * 50)
    
    print(results_df)

    # --- 5. CONCLUSÃO E RECOMENDAÇÃO ---
    top_features = results_df.index[:5].tolist()
    print("\n" + "="*50)
    print("                 RECOMENDAÇÃO")
    print("="*50)
    print("Com base na análise combinada, as features que consistentemente pontuam alto")
    print("em múltiplos testes são as candidatas mais fortes.")
    print("\nRECOMENDAÇÃO INICIAL: Considere usar o seguinte subconjunto de features")
    print("para o seu próximo experimento de modelo:")
    print(top_features)
    print("\n--- ANÁLISE CONCLUÍDA ---")