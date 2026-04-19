# Modelos/3GRU.py
"""
Define a arquitetura de um modelo GRU (Gated Recurrent Unit).
"""
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, dropout):
        """
        Inicializa as camadas do modelo GRU.
        
        Args:
            input_size (int): Número de features de entrada (ex: potência, vento, etc.).
            hidden_size (int): Número de neurônios na camada oculta da GRU.
            n_layers (int): Número de camadas da GRU empilhadas.
            output_size (int): Número de passos de tempo a prever (nosso horizonte).
            dropout (float): Probabilidade de dropout para regularização.
        """
        super(GRUModel, self).__init__()
        
        # Camada GRU: processa as sequências de entrada.
        # batch_first=True é crucial, pois nossos dados estão no formato (batch, seq, features).
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Camada linear: Mapeia a saída da GRU para o tamanho da nossa previsão.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Define a passagem dos dados pelo modelo (forward pass).
        
        Args:
            x (torch.Tensor): Tensor de entrada com shape (batch_size, seq_length, input_size).
        """
        # A saída da GRU contém as saídas de cada passo de tempo.
        # h_n é o estado oculto final.
        gru_out, h_n = self.gru(x)
        
        # Queremos usar a saída do último passo de tempo da sequência para fazer a previsão.
        # `gru_out[:, -1, :]` seleciona o último passo de tempo de cada sequência no batch.
        last_time_step_out = gru_out[:, -1, :]
        
        # Passa a saída do último passo pela camada linear para obter a previsão final.
        out = self.fc(last_time_step_out)
        
        return out