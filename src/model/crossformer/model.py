import torch
import torch.nn as nn
import math


class Crossformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, block_size=4, nhead=4, num_layers=2, dropout=0.1, output_window=5):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        self.output_window = output_window  # <<<< Numero di passi temporali da predire

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.aggregator = nn.AdaptiveAvgPool1d(1)  # aggrega globalmente il blocco
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        x: [B, T, F]  (batch, time, features)
        output: [B, output_window, output_dim]
        """
        B, T, F = x.shape
        assert T % self.block_size == 0, "Sequence length must be divisible by block size"

        # Proiezione + Positional Encoding
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        # Divisione in blocchi temporali
        n_blocks = T // self.block_size
        blocks = x.view(B, n_blocks, self.block_size, self.d_model)
        blocks = blocks.reshape(-1, self.block_size, self.d_model)  # [B * n_blocks, block_size, d_model]
        # Self-attention all'interno dei blocchi
        encoded_blocks = self.encoder(blocks)

        # Aggregazione (es. media) su ciascun blocco
        agg = self.aggregator(encoded_blocks.transpose(1, 2)).squeeze(-1)  # [B * n_blocks, d_model]
        agg = agg.view(B, n_blocks, self.d_model)  # [B, n_blocks, d_model]
        # Secondo livello di attenzione tra i blocchi
        global_encoded = self.encoder(agg)  # [B, n_blocks, d_model]

        out = self.output_proj(global_encoded)  # [B, n_blocks, output_dim]
        # Prendi solo gli ultimi `output_window` blocchi
        out = out[:, -self.output_window:, :]  # [B, output_window, output_dim]

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
