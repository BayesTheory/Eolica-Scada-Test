# Salve como utils_modelos.py

import mlflow
import mlflow.pytorch
import pickle
import os

def carregar_especialista(model_name: str, mlflow_uri: str = "http://localhost:5000"):
    """
    Carrega um modelo especialista (Detector ou Forecaster) e seu scaler
    a partir do registro do MLflow.
    """
    print(f"Carregando especialista '{model_name}' do MLflow em {mlflow_uri}...")
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.pytorch.load_model(model_uri)
        
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(name=model_name, stages=["None"])[0]
        run_id = model_version.run_id
        
        # Cria um diretório de artefatos dentro da pasta do co-piloto
        local_dir = f"mlflow_artifacts_{model_name}"
        if not os.path.exists(local_dir): os.makedirs(local_dir)
            
        # O download é relativo ao diretório de execução atual
        local_path = client.download_artifacts(run_id, ".", local_dir)
        
        scaler_path = None
        for file in os.listdir(local_path):
            if file.endswith(".pkl"):
                scaler_path = os.path.join(local_path, file)
                break
        
        if not scaler_path:
            raise FileNotFoundError("Nenhum arquivo de scaler (.pkl) encontrado nos artefatos.")

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            
        print(f"-> Especialista '{model_name}' e scaler carregados com sucesso.")
        return model, scaler
    except Exception as e:
        print(f"\nERRO CRÍTICO ao carregar o especialista '{model_name}': {e}")
        return None, None