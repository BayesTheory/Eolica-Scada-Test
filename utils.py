"""
Módulo de funções auxiliares (ferramentas).
Contém a lógica para criar sequências (janelas) para modelos de série temporal.
"""
import numpy as np

def create_sequences(data, input_window_size, output_horizon_size):
    """
    Transforma um array 2D de séries temporais em um dataset de janelas 3D.
    Formato de saída: (amostras, passos_de_tempo, features)
    """
    X, y = [], []
    # O loop deve garantir que haja dados suficientes para a janela de entrada e o horizonte de saída
    for i in range(len(data) - input_window_size - output_horizon_size + 1):
        # Janela de entrada (passado)
        input_seq = data[i:(i + input_window_size)]
        X.append(input_seq)
        
        # Horizonte de previsão (futuro)
        output_seq = data[(i + input_window_size):(i + input_window_size + output_horizon_size)]
        y.append(output_seq)
        
    return np.array(X), np.array(y)