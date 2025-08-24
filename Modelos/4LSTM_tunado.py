"""
Define a arquitetura de um modelo LSTM "Tunado" (v4.5).

Esta arquitetura é significativamente mais profunda e robusta, projetada para
tarefas complexas de classificação de séries temporais, como a previsão de falhas.
Incorpora múltiplas camadas LSTM e uma "cabeça" de classificação MLP com
Batch Normalization e Dropout para maior poder de aprendizado e regularização.
"""
import torch
import torch.nn as nn

# NOME DA CLASSE CORRIGIDO
class LSTMTunadoModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, dropout):
        """
        Inicializa as camadas do modelo LSTM Tunado.
        """
        super(LSTMTunadoModel, self).__init__()
        
        # --- Bloco Recorrente (LSTM Profundo) ---
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # --- Cabeça de Classificação (MLP Robusto) ---
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        """
        Define a passagem dos dados pelo modelo (forward pass).
        """
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        logits = self.classification_head(last_time_step_out)
        return logits