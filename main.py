"""
Painel de Controle (v2.3) - Focado em Previsão de Séries Temporais.
Lê a configuração de um arquivo YAML e orquestra o pipeline de treinamento.
"""
import importlib
import yaml
import sys

# --- Seleção do Modelo ---
# Para executar, passe a chave do modelo como argumento na linha de comando.
# Exemplo: python main.py 3GRU
# Se nenhum argumento for passado, usará '3GRU' como padrão.
MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else '3GRU'

# ==============================================================================
# CARREGAMENTO DA CONFIGURAÇÃO
# ==============================================================================
try:
    with open('config.yaml', 'r') as f:
        config_geral = yaml.safe_load(f)
except FileNotFoundError:
    print("ERRO: Arquivo 'config.yaml' não encontrado. Certifique-se de que ele existe no mesmo diretório.")
    sys.exit(1)

# Extrai as configurações do arquivo YAML
INPUT_WINDOW_STEPS = config_geral['INPUT_WINDOW_STEPS']
OUTPUT_HORIZON_STEPS = config_geral['OUTPUT_HORIZON_STEPS']
CV_SPLITS = config_geral['CV_SPLITS']
CONFIG_MODELOS = config_geral['MODELOS']


# ==============================================================================
# EXECUÇÃO DO PIPELINE
# ==============================================================================

if __name__ == "__main__":
    print("--- INICIANDO EXECUÇÃO DO PIPELINE DE TREINAMENTO ---")

    if MODEL_KEY not in CONFIG_MODELOS:
        print(f"ERRO: Chave do modelo '{MODEL_KEY}' não encontrada no 'config.yaml'.")
        print(f"Modelos disponíveis: {list(CONFIG_MODELOS.keys())}")
        sys.exit(1)

    config_modelo = CONFIG_MODELOS[MODEL_KEY]
    model_type = config_modelo.get('type', 'pytorch')
    
    print(f"\nModelo Selecionado: {MODEL_KEY} (Tipo: {model_type})")

    try:
        module = importlib.import_module(config_modelo['module_name'])
        ModelClass = getattr(module, config_modelo['class_name'])
    except Exception as e:
        print(f"ERRO: Não foi possível carregar a classe do modelo: {e}")
        raise e

    if model_type == 'pytorch':
        from train import run_pytorch_cv
        run_pytorch_cv(
            model_class=ModelClass, model_key=MODEL_KEY, config=config_modelo,
            input_window_size=INPUT_WINDOW_STEPS,
            output_horizon_size=OUTPUT_HORIZON_STEPS, cv_splits=CV_SPLITS
        )
    elif model_type == 'statistical':
        # Nota: O ARIMA no config.yaml usa lista [5,1,0], convertemos para tupla aqui
        config_modelo['params']['order'] = tuple(config_modelo['params']['order'])
        from train import run_arima_cv
        run_arima_cv(
            model_class=ModelClass, model_key=MODEL_KEY, config=config_modelo,
            output_horizon_size=OUTPUT_HORIZON_STEPS, cv_splits=CV_SPLITS
        )
    elif model_type == 'tree_based':
        from train import run_xgboost_cv
        run_xgboost_cv(
            model_class=ModelClass, model_key=MODEL_KEY, config=config_modelo,
            cv_splits=CV_SPLITS
        )
    else:
        print(f"ERRO: Tipo de modelo '{model_type}' não suportado pelo pipeline.")

    print("\n--- PIPELINE DE TREINAMENTO FINALIZADO ---")