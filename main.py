"""
Painel de Controle (v2.9) - Ponto de Entrada Único do Pipeline

Este script orquestra todo o pipeline de MLOps com base em modos de operação.

Modos Disponíveis:
- train: Treina um modelo específico.
  Uso: python main.py train <MODEL_KEY>
  Ex:  python main.py train 4.5LSTM_tunado

- detect: Executa o detector de anomalias usando modelos em produção.
  Uso: python main.py detect

- report: Gera o relatório comparativo de desempenho dos modelos treinados.
  Uso: python main.py report
"""
import importlib
import yaml
import sys
import mlflow

# --- CONFIGURAÇÕES GLOBAIS ---
EXPERIMENT_NAME = "Wind Power Forecasting"

def load_config():
    """Carrega o arquivo de configuração YAML."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("ERRO: Arquivo 'config.yaml' não encontrado.")
        sys.exit(1)

def run_training_pipeline(model_key, config):
    """Orquestra o treinamento de um modelo específico."""
    print(f"--- INICIANDO MODO DE TREINAMENTO para o modelo: {model_key} ---")
    
    CONFIG_MODELOS = config['MODELOS']
    if model_key not in CONFIG_MODELOS:
        print(f"ERRO: Chave do modelo '{model_key}' não encontrada no 'config.yaml'.")
        print(f"Modelos disponíveis: {list(CONFIG_MODELOS.keys())}")
        return

    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow Experiment set to: '{EXPERIMENT_NAME}'")

    config_modelo = CONFIG_MODELOS[model_key]
    model_type = config_modelo.get('type')
    
    print(f"\nModelo Selecionado: {model_key} (Tipo: {model_type})")

    try:
        module = importlib.import_module(config_modelo['module_name'])
        ModelClass = getattr(module, config_modelo['class_name'])
    except Exception as e:
        print(f"ERRO: Não foi possível carregar a classe do modelo: {e}")
        raise e

    # Roteamento para os motores de treino
    if model_type == 'pytorch':
        from train import run_pytorch_cv
        run_pytorch_cv(ModelClass, model_key, config_modelo, config['INPUT_WINDOW_STEPS'], config['OUTPUT_HORIZON_STEPS'], config['CV_SPLITS'])
    elif model_type == 'pytorch_classifier':
        from train_fault_predictor import train_fault_predictor
        train_fault_predictor(ModelClass, model_key, config_modelo, config['INPUT_WINDOW_STEPS'], config['CV_SPLITS'])
    elif model_type == 'statistical':
        from train import run_arima_cv
        config_modelo['params']['order'] = tuple(config_modelo['params']['order'])
        run_arima_cv(ModelClass, model_key, config_modelo, config['OUTPUT_HORIZON_STEPS'], config['CV_SPLITS'])
    elif model_type == 'tree_based':
        from train import run_xgboost_cv
        run_xgboost_cv(ModelClass, model_key, config_modelo, config['CV_SPLITS'])
    else:
        print(f"ERRO: Tipo de modelo '{model_type}' não suportado.")
    
    print("\n--- PIPELINE DE TREINAMENTO FINALIZADO ---")

def run_detection_pipeline():
    """Executa a lógica de detecção de anomalias."""
    print("--- INICIANDO MODO DE DETECÇÃO DE ANOMALIAS ---")
    # Importamos a lógica aqui para não carregar desnecessariamente em outros modos
    from anomaly_detector import main as anomaly_main
    anomaly_main()
    print("\n--- DETECÇÃO DE ANOMALIAS FINALIZADA ---")

def run_report_pipeline():
    """Executa a geração do relatório comparativo."""
    print("--- INICIANDO MODO DE GERAÇÃO DE RELATÓRIO ---")
    from generate_comparison_report import generate_comparison_report
    generate_comparison_report()
    print("\n--- GERAÇÃO DE RELATÓRIO FINALIZADA ---")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERRO: Modo de operação não especificado.")
        print("Uso: python main.py [train|detect|report] [argumentos...]")
        sys.exit(1)

    mode = sys.argv[1]
    config = load_config()

    if mode == 'train':
        if len(sys.argv) < 3:
            print("ERRO: Chave do modelo não especificada para o modo 'train'.")
            print("Uso: python main.py train <MODEL_KEY>")
            sys.exit(1)
        model_key_to_train = sys.argv[2]
        run_training_pipeline(model_key_to_train, config)
    
    elif mode == 'detect':
        run_detection_pipeline()
    
    elif mode == 'report':
        run_report_pipeline()

    else:
        print(f"ERRO: Modo '{mode}' desconhecido.")
        print("Modos válidos: 'train', 'detect', 'report'.")