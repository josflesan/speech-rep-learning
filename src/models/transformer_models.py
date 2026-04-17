import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

### COMPONENTS ###

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class StochasticEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nheads: int, d_ff: int, stoch_prob: float, device: str, dropout: float = 0.0):
        super().__init__()
        self.device = device
        self.self_attn = nn.MultiheadAttention(d_model, nheads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.stoch_prob = stoch_prob

    def forward(self, x, mask):
        layers_dropped = 0
        source_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(self.device)
        if self.training:
            noise = torch.empty([1], dtype=torch.long, device=self.device)
            noise = noise.bernoulli_(self.stoch_prob)

            if noise.item() == 1.0:
                attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask, attn_mask=source_mask)
                x = self.norm1(x + self.dropout(attn_output))
            else:
                layers_dropped += 1
                x = self.norm1(x)

        ff_output = self.feed_forward(x)
        if self.training:
            noise = torch.empty([1], dtype=torch.long, device=self.device)
            noise = noise.bernoulli_(self.stoch_prob)

            if noise.item() == 1.0:
                ff_output = self.feed_forward(x)
                x = self.norm2(x + self.dropout(ff_output))
            else:
                layers_dropped += 1
                x = self.norm2(x)

        return x, layers_dropped

### PRETRAINED APC NETWORKS ###

class ApcStochasticTransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead, stoch_prob, linear_decay=False, device='cpu'):
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_size
        self.stoch_prob = stoch_prob
        self.linear_decay = linear_decay

        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)

        # Create a list to hold the transformer encoder layers
        self.encoder_layers = []
        for i in range(num_layers):
            stochastic_prob = 1 - (i / num_layers) * (1 - self.stoch_prob) if self.linear_decay else self.stoch_prob
            self.encoder_layers.append(
                StochasticEncoderLayer(
                    d_model=hidden_size,
                    nheads=nhead,
                    d_ff=hidden_size*4,
                    stoch_prob=stochastic_prob,
                    device=self.device,
                    dropout=0.0
                )
            )
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, feat, mask=None):
        # Embedding and positional encoding
        feat = self.embedding(feat)
        feat = self.pos_encoder(feat)

        # Generate source mask
        feat = feat.permute(1, 0, 2)

        if mask is not None:
            mask = mask.to(self.device)

        # Process transformer encoder layers
        total_layers_dropped = 0
        for layer in self.encoder_layers:
            feat, layers_dropped = layer(feat, mask)
            feat = feat.to(self.device)

            total_layers_dropped += layers_dropped

        transformer_out = feat.permute(1, 0, 2)

        # Project acoustic features
        predicted_mel = self.proj(transformer_out)

        print(f"Skip Connections Dropped: {total_layers_dropped}/{len(self.encoder_layers) * 2}")

        return predicted_mel, transformer_out

class ApcTransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead, dropout=0.0, device='cpu'):
        super().__init__()

        self.hidden_dim = hidden_size
        self.device = device

        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.embedding = nn.Linear(input_size, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=hidden_size*4,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, feat, mask=None):
        # Embedding layer
        feat = self.embedding(feat)
        feat = self.pos_encoder(feat)

        # Generate source mask
        source_mask = nn.Transformer.generate_square_subsequent_mask(feat.size(1)).to(self.device)

        if mask is not None:
            mask = mask.to(self.device)

        # Transformer encoder layer
        feat = feat.permute(1, 0, 2)
        transformer_out = self.encoder(feat, mask=source_mask, src_key_padding_mask=mask)
        transformer_out = transformer_out.permute(1, 0, 2)

        # Project acoustic features
        predicted_mel = self.proj(transformer_out)

        return predicted_mel, transformer_out

