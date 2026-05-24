from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 3
    model_type: str = "gru"
    transformer_heads: int = 4
    transformer_ff_dim: int = 256


class BullBearGRU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        gru_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x)
        last_hidden = hidden[-1]
        return self.head(last_hidden)


class BullBearLSTM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        last_hidden = hidden[-1]
        return self.head(last_hidden)


class BullBearDirectClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.input_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x[:, -1, :])


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        encoding = torch.zeros(max_len, hidden_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len]


class BullBearTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.transformer_heads != 0:
            raise ValueError(
                f"hidden_dim={config.hidden_dim} must be divisible by transformer_heads={config.transformer_heads}."
            )
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x)
        hidden = self.positional_encoding(hidden)
        encoded = self.encoder(hidden)
        return self.head(encoded[:, -1, :])


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class BullBearTCN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = config.input_dim
        for layer_idx in range(config.num_layers):
            dilation = 2**layer_idx
            layers.append(
                TemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=config.hidden_dim,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=config.dropout,
                )
            )
            in_channels = config.hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        return self.head(encoded[:, -1, :])


def build_model(config: ModelConfig) -> nn.Module:
    if config.model_type == "gru":
        return BullBearGRU(config)
    if config.model_type == "lstm":
        return BullBearLSTM(config)
    if config.model_type == "tcn":
        return BullBearTCN(config)
    if config.model_type == "transformer":
        return BullBearTransformer(config)
    if config.model_type == "direct":
        return BullBearDirectClassifier(config)
    raise ValueError(f"Unsupported model_type: {config.model_type}")
