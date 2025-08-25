"""
Painel de Controle (v2.0) - Orquestrador Final

Este script orquestra todo o pipeline de ML, chamando os módulos
de processamento de dados e treinamento de modelos.
"""
import yaml
import sys
import mlflow

# Importa os módulos unificados
import pipeline_data
import train_models

def load_config():
    """Carrega o arquivo de configuração do projeto."""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("ERRO: Arquivo 'config.yaml' não encontrado.")
        sys.exit(1)
    except Exception as e:
        print(f"ERRO ao ler o arquivo 'config.yaml'. Detalhe: {e}")
        sys.exit(1)

def run_pipeline(mode: str, mission_key: str = None, model_key: str = None):
    """
    Executa o pipeline de acordo com o modo de operação e a missão especificada.
    """
    config = load_config()
    
    # Inicia o MLflow no diretório do projeto
    mlflow.set_tracking_uri(config.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))

    if mode == 'process_data':
        print("\n--- INICIANDO PROCESSAMENTO DE DADOS ---")
        pipeline_data.main()
        print("--- PROCESSAMENTO DE DADOS CONCLUÍDO ---")
        
    elif mode == 'train':
        if mission_key and model_key:
            print(f"\n--- INICIANDO TREINAMENTO PARA A MISSÃO '{mission_key}' ---")
            train_models.run_training_mission(mission_key, model_key, config)
            print("--- TREINAMENTO CONCLUÍDO ---")
        else:
            print("ERRO: Para o modo 'train', você deve especificar a missão e o modelo.")
            print("Uso: python main.py train <nome_da_missao> <nome_do_modelo>")
    
    else:
        print(f"ERRO: Modo '{mode}' desconhecido. Use 'process_data' ou 'train'.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        mission_key = sys.argv[2] if len(sys.argv) > 2 else None
        model_key = sys.argv[3] if len(sys.argv) > 3 else None
        run_pipeline(mode, mission_key, model_key)
    else:
        print("ERRO: Modo de operação não especificado. Use 'process_data' ou 'train'.")