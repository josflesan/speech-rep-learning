import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

### PRETRAINED APC NETWORKS ###

class UniLstmStochastic(nn.Module):
    """LSTM with Stochastic Depth"""

    def __init__(self, input_size, hidden_size, num_layers, stochastic_depth_prob=0.5, linear=False):
        super().__init__()

        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size, hidden_size, num_layers=1, batch_first=True)
            for i in range(num_layers)
        ])

        self.proj = nn.Linear(hidden_size, input_size)
        self.stochastic_depth_prob = stochastic_depth_prob
        self.linear = linear  # Flag that determines whether we use linear decay

    def forward(self, feat, lengths):
        packed_feat = pack_padded_sequence(feat, lengths, True)

        num_layers_passed = 0
        for i, lstm_layer in enumerate(self.lstm_layers):
            unpacked_feat, _ = pad_packed_sequence(packed_feat, True)

            drop_prob = 1 - (i / len(self.lstm_layers)) * (1 - self.stochastic_depth_prob) if self.linear else self.stochastic_depth_prob
            
            if self.training and torch.rand(1).item() < drop_prob:
                lstm_out = unpacked_feat
            else:
                num_layers_passed += 1
                
                packed_lstm_out, _ = lstm_layer(packed_feat)
                lstm_out, _ = pad_packed_sequence(packed_lstm_out, True)

                if unpacked.feat.size(-1) == lstm_out.size(-1):
                    lstm_out = unpacked_feat + lstm_out

            packed_feat = pack_padded_sequence(lstm_out, lengths, True)

        print(f"Number of layers passed: {num_layers_passed}/{len(self.lstm_layers)}")

        # Project to predict Mel features
        predicted_mel = self.proj(lstm_out)

        return predicted_mel, lstm_out

class UniLstmSkip(nn.Module):
    """Baseline LSTM with skip connections"""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()

        # Create a ModuleList to store the LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size,
                    hidden_size,
                    num_layers=1,
                    dropout=dropout,
                    batch_first=True)
            for i in range(num_layers)
        ])

        # Fully connected layer for output
        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, feat, lengths):
        packed_feat = pack_padded_sequence(feat, lengths, True)

        # Forward pass through LSTM layers with skip connections
        for lstm_layer in self.lstm_layers:
            packed_lstm_out, _ = lstm_layer(packed_feat)
            lstm_out, _ = pad_packed_sequence(packed_lstm_out, True)

            unpacked_feat, _ = pad_packed_sequence(packed_feat, True)

            # Add skip connection to next input
            if unpacked_feat.size(-1) == lstm_out.size(-1):
                lstm_out = unpacked_feat + lstm_out

            packed_feat = pack_padded_sequence(lstm_out, lengths, True)

        # Project to predict Mel features
        predicted_mel = self.proj(lstm_out)
        
        return predicted_mel, lstm_out

class UniLstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout,
            bidirectional=False, batch_first=True)
        self.proj = torch.nn.Linear(hidden_size, input_size)

    def forward(self, feat, lengths):
        packed_feat = pack_padded_sequence(feat, lengths, True)
        packed_lstm_out, _ = self.lstm(packed_feat)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, True)

        # Project to predict Mel features
        predicted_mel = self.proj(lstm_out)

        return predicted_mel, lstm_out

class FractalLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, n_cols, drop_path_prob=0.15, dropout=0.0):
        super().__init__()
        self.n_cols = n_cols
        self.dropout = nn.Dropout(dropout)
        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(n_cols)])
        self.max_depth = 2 ** (n_cols - 1)
        self.drop_path_prob = drop_path_prob

        dist = self.max_depth
        self.count = [0] * self.max_depth

        for col in self.columns:
            for i in range(self.max_depth):
                if (i + 1) % dist == 0:
                    if self.count[i] != 0:
                        input_size = hidden_size

                    module = nn.LSTM(input_size, hidden_size, 1, dropout=dropout, 
                        bidirectional=False, batch_first=True)
                    self.count[i] += 1
                else:
                    module = None

                col.append(module)

            dist //= 2

    def forward(self, feat, lengths):
        # Generate a random number, if less than 15%, randomly pick a column to drop
        columnDropped = -1
        if torch.rand(1).item() < self.drop_path_prob:
            columnDropped = random.choice([0, 1])

        out = [feat for _ in range(self.n_cols)]
        for i in range(self.max_depth):
            st = self.n_cols - self.count[i]
            cur_outs = []
            for c in range(st, self.n_cols):
                if columnDropped == c:
                    continue

                cur_in = out[c]
                cur_module = self.columns[c][i]

                # Pack input and feed through LSTM layer
                packed_feat = pack_padded_sequence(cur_in, lengths, True)
                packed_out, _ = cur_module(packed_feat)
                # Unpack output of LSTM layers
                lstm_out, _ = pad_packed_sequence(packed_out, True)

                cur_outs.append(lstm_out)
            
            if len(cur_outs):
                n_out = torch.stack(cur_outs)
                n_out = n_out.mean(dim=0)  # Average over all the columns

                for c in range(st, self.n_cols):
                    out[c] = n_out

        return self.dropout(out[-1])
    
class FractalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_cols, n_layer, device='cpu'):
        super().__init__()

        self.device = device
        self.blocks = nn.ModuleList([FractalLSTMBlock(input_size if i == 0 else hidden_size, hidden_size, n_cols) for i in range(n_layer)])
        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, feat, lengths):
        for block in self.blocks:
            lstm_out = block(feat, lengths)
            feat = lstm_out

        predicted_mel = self.proj(lstm_out)

        return predicted_mel, lstm_out

### PROBING NETWORKS ###
    
class ProbeLSTM(nn.Module):
    def __init__(self, pretrained_lstm, hidden_size, num_classes):
        super().__init__()
        self.lstm = pretrained_lstm
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # Forward pass through the LSTM
        _, hidden = self.lstm(x, lengths)

        # Project the output using a linear layer
        output = self.fc(hidden)

        return hidden, output

