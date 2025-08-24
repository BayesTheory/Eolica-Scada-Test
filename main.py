"""
Painel de Controle (v4.1) - Orquestrador Baseado em Missões
- CORRIGIDO: Abre o config.yaml com codificação UTF-8 para evitar UnicodeDecodeError.
"""
import importlib
import yaml
import sys
import mlflow
from constants import MLFLOW_TRACKING_URI

def load_config():
    """Carrega o config.yaml de forma segura, especificando a codificação UTF-8."""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("ERRO: Arquivo 'config.yaml' não encontrado.")
        sys.exit(1)
    except Exception as e:
        print(f"ERRO ao ler o arquivo 'config.yaml'. Verifique a formatação e a codificação.")
        print(f"Detalhe: {e}")
        sys.exit(1)

# ... (O resto do seu main.py permanece exatamente o mesmo) ...

def run_training_pipeline(mission_key, model_key, config):
    print(f"--- INICIANDO MISSÃO DE TREINAMENTO '{mission_key}' PARA O MODELO '{model_key}' ---")
    
    if mission_key not in config.get('MISSIONS', {}):
        print(f"ERRO: Missão '{mission_key}' não encontrada na seção 'MISSIONS' do config.yaml.")
        return
    mission_config = config['MISSIONS'][mission_key]
    
    if model_key not in mission_config.get('model_keys_to_run', []):
        print(f"ERRO: Modelo '{model_key}' não está listado para a missão '{mission_key}'.")
        return
    if model_key not in config.get('MODEL_DEFINITIONS', {}):
        print(f"ERRO: Definição do modelo '{model_key}' não encontrada em 'MODEL_DEFINITIONS'.")
        return

    model_def = config['MODEL_DEFINITIONS'][model_key]
    model_mission_params = mission_config['model_params'][model_key]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(mission_config['experiment_name'])
    print(f"MLflow Experiment set to: '{mission_config['experiment_name']}'")

    try:
        module = importlib.import_module(model_def['module_name'])
        ModelClass = getattr(module, model_def['class_name'])
    except Exception as e:
        print(f"ERRO: Não foi possível carregar a classe do modelo: {e}")
        raise e

    trainer_script_name = mission_config.get('trainer_script')
    if not trainer_script_name:
        print(f"ERRO: 'trainer_script' não definido para a missão '{mission_key}' no config.yaml.")
        return
        
    print(f"Roteando para o motor de treino: '{trainer_script_name}'...")
    
    if trainer_script_name == 'train_forecasting_model':
        from train_forecasting_model import train_model
        train_model(ModelClass, model_key, model_mission_params, config)
    elif trainer_script_name == 'train_anomaly_model':
        from train_anomaly_model import train_anomaly_specialist
        train_anomaly_specialist(ModelClass, model_key, model_mission_params, mission_config, config)
    else:
        print(f"ERRO: Motor de treino '{trainer_script_name}' desconhecido.")
    print("\n--- MISSÃO DE TREINAMENTO FINALIZADA ---")

def run_detection_pipeline():
    print("--- INICIANDO DETECÇÃO DE ANOMALIAS ---")
    from detect_anomalies import detect 
    config = load_config()
    detect(config)
    print("\n--- DETECÇÃO DE ANOMALIAS FINALIZADA ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERRO: Modo de operação não especificado. Use 'train' ou 'detect'.")
        sys.exit(1)
    mode = sys.argv[1]
    config = load_config()
    if mode == 'train':
        if len(sys.argv) < 4:
            print("ERRO: Uso: python main.py train <NOME_DA_MISSAO> <CHAVE_DO_MODELO>")
            sys.exit(1)
        mission_to_run = sys.argv[2]
        model_to_train = sys.argv[3]
        run_training_pipeline(mission_to_run, model_to_train, config)
    elif mode == 'detect':
        run_detection_pipeline()
    else:
        print(f"ERRO: Modo '{mode}' desconhecido.")