# model.py
import torch
import torch.nn as nn
import yaml

class myTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, input_window, output_window,
                 d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_window = input_window
        self.output_window = output_window

        # Proietta le feature di input nello spazio d_model
        self.input_linear = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Mapper finale da T_in → T_out
        # Flatten: [B, T_in, d_model] → [B, T_in * d_model]
        # Poi MLP → [B, T_out * output_dim] → reshape [B, T_out, output_dim]
        self.fc = nn.Sequential(
            nn.Flatten(),  # [B, T_in * d_model]
            nn.Linear(input_window * d_model, output_window * output_dim)
        )
        self.output_dim = output_dim

    def forward(self, src):
        """
        src: [B, T_in, input_dim]
        returns: [B, T_out, output_dim]
        """
        x = self.input_linear(src)
        x = self.pos_encoder(x)
        x = self.encoder(x)  # [B, T_in, d_model]
        x = self.fc(x)       # [B, T_out * output_dim]
        x = x.view(-1, self.output_window, self.output_dim)  # [B, T_out, output_dim]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)