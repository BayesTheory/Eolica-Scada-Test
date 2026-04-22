"""
Painel de Controle (v2.8) - Orquestrador Principal do Pipeline de ML

Responsabilidades:
1.  Define um nome de experimento centralizado para agrupar as execuções no MLflow.
2.  Lê a configuração completa do projeto a partir do 'config.yaml'.
3.  Recebe a chave do modelo a ser treinado como um argumento de linha de comando.
4.  Carrega dinamicamente a classe do modelo correspondente.
5.  Roteia a execução para o motor de treinamento apropriado:
    - `train.py` para modelos de regressão (previsão de potência).
    - `train_fault_predictor.py` para modelos de classificação (previsão de falhas).
"""
import importlib
import yaml
import sys
import mlflow

# --- NOME DO EXPERIMENTO NO MLFLOW ---
# Todas as execuções serão agrupadas sob este nome na UI do MLflow.
EXPERIMENT_NAME = "Wind Power Forecasting"

# --- Seleção do Modelo ---
# Para executar, passe a chave do modelo como argumento na linha de comando.
# Exemplo: python main.py 4.5LSTM_tunado
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

# Extrai as configurações globais do arquivo YAML
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

    # --- DEFININDO O EXPERIMENTO ATIVO NO MLFLOW ---
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow Experiment set to: '{EXPERIMENT_NAME}'")

    config_modelo = CONFIG_MODELOS[MODEL_KEY]
    model_type = config_modelo.get('type', 'pytorch')
    
    print(f"\nModelo Selecionado: {MODEL_KEY} (Tipo: {model_type})")

    try:
        module = importlib.import_module(config_modelo['module_name'])
        ModelClass = getattr(module, config_modelo['class_name'])
    except Exception as e:
        print(f"ERRO: Não foi possível carregar a classe do modelo: {e}")
        raise e

    # --- LÓGICA DE ROTEAMENTO PARA OS MOTORES DE TREINO ---
    if model_type == 'pytorch':
        from train import run_pytorch_cv
        run_pytorch_cv(
            model_class=ModelClass, model_key=MODEL_KEY, config=config_modelo,
            input_window_size=INPUT_WINDOW_STEPS, output_horizon_size=OUTPUT_HORIZON_STEPS, cv_splits=CV_SPLITS
        )
    elif model_type == 'pytorch_classifier':
        from train_fault_predictor import train_fault_predictor
        train_fault_predictor(
            model_class=ModelClass, model_key=MODEL_KEY, config=config_modelo,
            input_window_size=INPUT_WINDOW_STEPS, cv_splits=CV_SPLITS
        )
    elif model_type == 'statistical':
        from train import run_arima_cv
        config_modelo['params']['order'] = tuple(config_modelo['params']['order'])
        run_arima_cv(
            model_class=ModelClass, model_key=MODEL_KEY, config=config_modelo,
            output_horizon_size=OUTPUT_HORIZON_STEPS, cv_splits=CV_SPLITS
        )
    elif model_type == 'tree_based':
        from train import run_xgboost_cv
        run_xgboost_cv(
            model_class=ModelClass, model_key=MODEL_KEY, config=config_modelo, cv_splits=CV_SPLITS
        )
    else:
        print(f"ERRO: Tipo de modelo '{model_type}' não suportado pelo pipeline.")

    print("\n--- PIPELINE DE TREINAMENTO FINALIZADO ---")