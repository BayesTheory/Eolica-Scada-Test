# Modelos/4LSTM.py
"""
Define a arquitetura de um modelo LSTM (Long Short-Term Memory).
"""
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, dropout):
        """
        Inicializa as camadas do modelo LSTM.
        Os argumentos são os mesmos do modelo GRU.
        """
        super(LSTMModel, self).__init__()
        
        # A única mudança real é aqui: de nn.GRU para nn.LSTM.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # A camada de saída é a mesma.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Define a passagem dos dados pelo modelo (forward pass).
        """
        # A lógica do forward pass é idêntica à da GRU.
        # A LSTM retorna a saída e uma tupla (estado oculto, estado da célula).
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Pegamos a saída do último passo de tempo.
        last_time_step_out = lstm_out[:, -1, :]
        
        # E passamos pela camada linear.
        out = self.fc(last_time_step_out)
        
        return out