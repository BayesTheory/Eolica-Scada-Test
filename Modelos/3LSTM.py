"""
Definição do modelo LSTM com Batch Normalization para maior estabilidade.
"""
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # --- NOVA CAMADA DE BATCH NORM NA ENTRADA ---
        # Normaliza as features de entrada antes de passá-las para a LSTM.
        # O shape esperado é (N, C, L) ou (N, L), então usamos para (N, C)
        # onde C é o número de features.
        self.batch_norm_input = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,  # Espera input no formato (batch, seq_len, features)
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x tem shape: (batch_size, seq_len, input_size)
        
        # Para aplicar BatchNorm1d, precisamos tratar as features ao longo do tempo.
        # Permutamos para (batch_size, input_size, seq_len)
        x_permuted = x.permute(0, 2, 1)
        x_norm = self.batch_norm_input(x_permuted)
        
        # Voltamos para o formato original para a LSTM
        x_norm_permuted = x_norm.permute(0, 2, 1)

        # Inicializa os estados ocultos
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Passa os dados normalizados pela LSTM
        out, _ = self.lstm(x_norm_permuted, (h0, c0))
        
        # Pega a saída do último passo de tempo
        out = self.fc(out[:, -1, :])
        return out