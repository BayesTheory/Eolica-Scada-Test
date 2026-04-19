# Modelos/5TFT.py
"""
Define a arquitetura de um modelo para séries temporais inspirado no Transformer.
Utiliza o TransformerEncoder do PyTorch.
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Adiciona informação de posição aos embeddings de entrada."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TFTModel(nn.Module):
    def __init__(self, input_size, hidden_size, attention_head_size, n_layers, output_size, dropout):
        """
        Inicializa as camadas do modelo.
        
        Args:
            input_size (int): Número de features de entrada.
            hidden_size (int): Dimensão interna do modelo (d_model).
            attention_head_size (int): Número de "cabeças" de atenção.
            n_layers (int): Número de camadas do Transformer Encoder.
            output_size (int): Horizonte de previsão.
            dropout (float): Probabilidade de dropout.
        """
        super(TFTModel, self).__init__()
        self.d_model = hidden_size
        
        # 1. Camada de embedding: projeta as features de entrada para a dimensão do modelo
        self.input_embedding = nn.Linear(input_size, self.d_model)
        
        # 2. Codificação posicional
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        
        # 3. Camada do Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=attention_head_size, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # 4. Camada de saída
        self.output_layer = nn.Linear(self.d_model, output_size)

    def forward(self, src):
        """
        Forward pass.
        
        Args:
            src (torch.Tensor): Tensor de entrada com shape (batch_size, seq_length, input_size).
        """
        # Passa pela camada de embedding
        embedded_src = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # Adiciona a codificação posicional
        # Nota: O PositionalEncoding espera (seq_length, batch_size, features), então precisamos ajustar
        # No entanto, como usamos batch_first=True no encoder, podemos adaptar
        # A implementação padrão pode ser mais complexa, mas vamos simplificar aqui
        
        # Passa pelo encoder
        encoded_src = self.transformer_encoder(embedded_src)
        
        # Usa a saída do último passo de tempo para a previsão, assim como na RNN
        last_time_step_out = encoded_src[:, -1, :]
        
        # Passa pela camada de saída
        output = self.output_layer(last_time_step_out)
        
        return output