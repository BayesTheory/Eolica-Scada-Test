"""
Script de Inferência.
Carrega um modelo em produção do MLflow Model Registry e o utiliza para
fazer uma previsão em novos dados.
"""
import mlflow
import pandas as pd
import numpy as np
import pickle

# Configure este URI para apontar para o seu servidor MLflow
# Se estiver rodando localmente, execute `mlflow ui` no terminal e use o endereço
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# Se não estiver usando um servidor, comente a linha acima e o MLflow usará o diretório local `mlruns`

def load_model_from_registry(model_name, stage="Production"):
    """Carrega o modelo e os scalers associados do MLflow Registry."""
    try:
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Carrega o modelo PyFunc (um wrapper genérico que sabe como executar o modelo)
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
        
        # Para carregar os scalers, precisamos encontrar o run original que gerou esta versão do modelo
        client = mlflow.tracking.MlflowClient()
        model_version_details = client.get_latest_versions(name=model_name, stages=[stage])[0]
        run_id = model_version_details.run_id
        
        # Baixa os artefatos do scaler do run original
        local_path = client.download_artifacts(run_id, "scalers", ".")
        with open(f"{local_path}/scaler_features.pkl", "rb") as f:
            scaler_features = pickle.load(f)
        with open(f"{local_path}/scaler_target.pkl", "rb") as f:
            scaler_target = pickle.load(f)
            
        print("Modelo e scalers carregados com sucesso.")
        return model, scaler_features, scaler_target
        
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo '{model_name}' no estágio '{stage}'.")
        print(f"Verifique se o modelo foi treinado, registrado e promovido para o estágio correto no MLflow UI.")
        print(f"Detalhe do erro: {e}")
        return None, None, None


if __name__ == "__main__":
    MODEL_NAME = "wind-power-forecaster"
    STAGE = "Production" # Ou "Staging", "None", etc.
    
    print(f"Carregando modelo '{MODEL_NAME}' em estágio '{STAGE}'...")
    model, scaler_features, scaler_target = load_model_from_registry(MODEL_NAME, STAGE)

    if model:
        # --- SIMULAÇÃO DE NOVOS DADOS ---
        # Na vida real, estes seriam os últimos dados coletados da turbina.
        # A entrada precisa ter o formato (input_window_size, n_features)
        
        # Carregamos a config para pegar os tamanhos de janela e features
        import yaml
        with open('config.yaml', 'r') as f:
            config_geral = yaml.safe_load(f)
        
        INPUT_WINDOW_STEPS = config_geral['INPUT_WINDOW_STEPS']
        # Usaremos as features do modelo GRU como exemplo
        FEATURES = config_geral['MODELOS']['3GRU']['features']
        N_FEATURES = len(FEATURES)
        
        print(f"\nSimulando {INPUT_WINDOW_STEPS} amostras de dados de entrada com {N_FEATURES} features.")
        # Dados aleatórios com a mesma escala dos dados originais (aproximado)
        sample_raw_data = np.random.rand(INPUT_WINDOW_STEPS, N_FEATURES) * np.array([3000, 25, 20, 80])

        # 1. Normalizar os dados de entrada com o scaler CARREGADO
        input_data_scaled = scaler_features.transform(sample_raw_data)

        # 2. Formatar os dados para o que o modelo espera
        # O modelo pyfunc do MLflow com assinatura geralmente espera um array numpy ou DataFrame.
        # Para um modelo de sequência, precisamos ter certeza que o formato está correto.
        # A assinatura espera (1, 144, 4), mas pyfunc pode precisar de um formato 2D.
        # Vamos passar como numpy array com o batch de 1.
        input_tensor = np.array([input_data_scaled], dtype=np.float32)

        # 3. Fazer a previsão
        print("Fazendo a previsão...")
        prediction_scaled = model.predict(input_tensor)

        # 4. Inverter a escala da previsão para obter o valor real em kW
        prediction_real = scaler_target.inverse_transform(prediction_scaled)

        # 5. Apresentar o resultado
        print("\n--- Resultado da Previsão ---")
        print(f"Previsão de Potência para os próximos {prediction_real.shape[1] * 10} minutos:")
        for i, val in enumerate(prediction_real[0]):
            print(f"  - T+{10*(i+1)} min: {val:.2f} kW")