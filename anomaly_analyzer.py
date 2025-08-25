"""
Módulo de Lógica de Negócio para Análise de Anomalias.
Encapsula o modelo e a lógica de inferência, tornando-os testáveis e
independentes do framework da API.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch
import pickle
import os

# ==============================================================================
# CLASSES AUXILIARES (do seu INFERENCIA.py)
# ==============================================================================

class TimeSeriesInferenceDataset(Dataset):
    """Dataset customizado para inferência eficiente."""
    def __init__(self, data, window_size):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.num_samples = len(self.data) - self.window_size + 1
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.data[idx : idx + self.window_size]

# ==============================================================================
# CLASSE PRINCIPAL: O CÉREBRO DA ANÁLISE DE ANOMALIAS
# ==============================================================================

class AnomalyAnalyzer:
    """
    Classe para carregar, gerenciar e executar a inferência do modelo
    de detecção de anomalias.
    """
    def __init__(self, model_name: str, mlflow_uri: str):
        self.model_name = model_name
        self.mlflow_uri = mlflow_uri
        self.model = None
        self.scaler = None
        self.threshold = None
        self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        """Carrega o modelo e o scaler do MLflow."""
        print(f"Carregando modelo '{self.model_name}'...")
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            model_uri = f"models:/{self.model_name}/latest"
            self.model = mlflow.pytorch.load_model(model_uri)
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(name=self.model_name, stages=["None"])[0]
            run_id = model_version.run_id
            
            local_dir = "mlflow_artifacts"
            if not os.path.exists(local_dir): os.makedirs(local_dir)
            local_path = client.download_artifacts(run_id, ".", local_dir)
            
            scaler_path = os.path.join(local_path, "health_model_scaler.pkl")
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"-> Modelo e scaler de anomalias carregados com sucesso.")
        except Exception as e:
            print(f"\nERRO CRÍTICO ao carregar o modelo de anomalias: {e}")
            self.model = None
            self.scaler = None

    def calculate_threshold(self, df_normal: pd.DataFrame, features: list, window_size: int):
        """Calcula o limiar de anomalia com base em dados normais."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Modelo ou scaler não carregado.")
        
        normal_data_scaled = self.scaler.transform(df_normal[features])
        normal_dataset = TimeSeriesInferenceDataset(normal_data_scaled, window_size)
        normal_loader = DataLoader(normal_dataset, batch_size=256, shuffle=False)
        
        errors_normal_list = []
        self.model.eval()
        with torch.no_grad():
            for seq_batch in normal_loader:
                reconstructed = self.model(seq_batch)
                error = torch.mean((seq_batch - reconstructed)**2, dim=(1, 2))
                errors_normal_list.append(error.cpu().numpy())
        
        errors_normal = np.concatenate(errors_normal_list)
        self.threshold = np.percentile(errors_normal, 99.5)
        print(f"-> Limiar de anomalia definido: {self.threshold:.6f}")
        return self.threshold

    def get_reconstruction_errors(self, data_df: pd.DataFrame, features: list, window_size: int) -> np.ndarray:
        """Calcula os erros de reconstrução para um dataframe de entrada."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Modelo ou scaler não carregado.")
            
        data_scaled = self.scaler.transform(data_df[features])
        dataset = TimeSeriesInferenceDataset(data_scaled, window_size)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        errors_list = []
        self.model.eval()
        with torch.no_grad():
            for seq_batch in loader:
                reconstructed = self.model(seq_batch)
                error = torch.mean((seq_batch - reconstructed)**2, dim=(1, 2))
                errors_list.append(error.cpu().numpy())
                
        return np.concatenate(errors_list)