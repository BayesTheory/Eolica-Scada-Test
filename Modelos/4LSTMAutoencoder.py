"""
Modelo LSTM Autoencoder com Arquitetura Encoder-Decoder (v2.0 - Final)

Esta é uma implementação robusta que força o modelo a aprender uma representação
latente (um "resumo") da sequência de entrada antes de reconstruí-la.
Esta arquitetura substitui a versão anterior Seq2Seq por ser mais eficaz
na detecção de anomalias estruturais nos dados.

- Encoder: Lê a sequência de entrada e a comprime em um único vetor de contexto
           (os estados finais 'hidden' e 'cell' da LSTM).
- Decoder: Recebe o vetor de contexto e o usa para gerar a sequência de saída,
           tentando reconstruir a entrada original.
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Codifica uma sequência de entrada em um vetor de contexto de tamanho fixo."""
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, n_layers, 
            dropout=dropout, batch_first=True
        )

    def forward(self, x):
        # Retorna apenas os estados finais 'hidden' e 'cell', que são o "resumo".
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    """Decodifica um vetor de contexto para reconstruir a sequência original."""
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, n_layers, 
            dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # Gera a saída para o passo de tempo atual e atualiza os estados.
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell

class LSTMAutoencoderModel(nn.Module):
    """
    A classe principal que une o Encoder e o Decoder.
    Esta é a classe que será importada e usada pelo motor de treino.
    """
    def __init__(self, input_size, hidden_size, n_layers, dropout, output_size):
        super(LSTMAutoencoderModel, self).__init__()
        
        # Garante que a saída do decoder corresponda à entrada original
        assert output_size == input_size, "Para um autoencoder, output_size deve ser igual a input_size"

        self.encoder = Encoder(input_size, hidden_size, n_layers, dropout)
        # O decoder recebe o estado oculto do encoder para reconstruir.
        # O input_size do decoder é o mesmo do encoder, pois usamos a saída de um passo
        # como entrada para o próximo (teacher forcing).
        self.decoder = Decoder(input_size, hidden_size, output_size, n_layers, dropout)

    def forward(self, x):
        # x tem shape [batch_size, sequence_length, num_features]
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        num_features = x.shape[2]
        device = x.device

        # 1. Codifica a entrada inteira em um "resumo" (vetor de contexto)
        hidden, cell = self.encoder(x)

        # 2. Prepara o tensor de saídas e a entrada inicial para o decoder
        outputs = torch.zeros(batch_size, sequence_length, num_features).to(device)
        
        # Usa o último passo de tempo da sequência de entrada como a "semente"
        # para o decoder começar a gerar a reconstrução.
        decoder_input = x[:, -1, :].unsqueeze(1)

        # 3. Gera a sequência de saída passo a passo, alimentando a saída anterior
        # como entrada para o próximo passo.
        for t in range(sequence_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            decoder_input = output # Teacher forcing

        return outputs