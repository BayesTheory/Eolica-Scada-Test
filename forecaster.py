# Salve este conteúdo como forecaster.py

"""
Módulo de Lógica de Negócio para Previsão de Potência.
Encapsula o modelo e a lógica de inferência para previsão.
"""
import pandas as pd
import numpy as np
import mlflow
import pickle
import os
import xgboost as xgb

# A função de criação de features agora está no módulo unificado 'train_models'
from train_models import _create_lagged_features_for_trees

# ==============================================================================
# CLASSE PRINCIPAL: O CÉREBRO DA PREVISÃO
# ==============================================================================

class Forecaster:
    """
    Classe para carregar, gerenciar e executar a inferência do modelo
    de previsão de potência.
    """
    def __init__(self, model_name: str, mlflow_uri: str):
        self.model_name = model_name
        self.mlflow_uri = mlflow_uri
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo do MLflow."""
        print(f"Carregando modelo '{self.model_name}'...")
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            model_uri = f"models:/{self.model_name}/latest"
            self.model = mlflow.xgboost.load_model(model_uri)
            print(f"-> Modelo de previsão carregado com sucesso.")
        except Exception as e:
            print(f"\nERRO CRÍTICO ao carregar o modelo de previsão: {e}")
            self.model = None

    def predict_next_step(self, data_df: pd.DataFrame, features: list, n_lags: int) -> float:
        """
        ### MÉTODO CORRIGIDO ###
        Prepara uma única amostra com os dados mais recentes e faz a previsão.
        """
        if self.model is None:
            raise RuntimeError("Modelo de previsão não carregado.")
        
        if len(data_df) < n_lags:
            raise ValueError(f"Não há dados suficientes para criar as features de lag. Mínimo: {n_lags}")
            
        # Pega a janela mais recente de dados
        last_window = data_df[features].iloc[-n_lags:]
        
        # Constrói a linha de features para a previsão
        feature_vector = {}
        for col in features:
            for lag in range(1, n_lags + 1):
                feature_name = f"{col}_lag_{lag}"
                # Pega o valor do passado correspondente
                feature_vector[feature_name] = [last_window[col].iloc[-lag]]

        # Cria um DataFrame de uma única linha
        X_pred = pd.DataFrame(feature_vector)
        
        # Garante que a ordem das colunas seja a mesma do treinamento
        X_pred = X_pred[self.model.feature_names_in_]

        # Faz a previsão
        prediction = self.model.predict(X_pred)
        
        return prediction[0]