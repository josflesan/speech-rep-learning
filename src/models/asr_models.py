import torch
import torch.nn as nn

class AsrEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_size=256):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.gru1 = nn.GRU(32 * 64, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.layernorm1 = nn.LayerNorm(hidden_size * 2)
        self.gru2 = nn.GRU(hidden_size * 2, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.layernorm2 = nn.LayerNorm(hidden_size * 2)
        self.gru3 = nn.GRU(hidden_size * 2, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.layernorm3 = nn.LayerNorm(hidden_size * 2)
        self.gru4 = nn.GRU(hidden_size * 2, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.layernorm4 = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension

        # Convolutional layers
        x = torch.relu(self.conv(x))
        x = torch.relu(self.conv2(x))

        # Reshape before feeding into GRU
        x = x.permute(2, 0, 1, 3)  # (batch_size, channels, length, features) -> (length, batch_size, channels, features)
        x = x.contiguous().view(x.size(0), x.size(1), -1) # Flatten the last two dimensions
        x = x.permute(1, 0, 2)  # (length, batch_size, channels*features) -> (batch_size, length, channels*features)

        # GRU layers
        x, _ = self.gru1(x)
        x = self.layernorm1(x)
        x, _ = self.gru2(x)
        x = self.layernorm2(x)
        x, _ = self.gru3(x)
        x = self.layernorm3(x)
        x, _ = self.gru4(x)
        x = self.layernorm4(x)

        # print(f"Encoder Output Shape: {x.shape}")

        return x

class AsrDecoder(nn.Module):
    def __init__(self, out_size, hidden_size=256):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, hidden):
        # Initialize input for decoding
        hidden = self.linear(hidden)  # Map from 512 (bidirectional) to 256 (unidirectional)        

        hidden, _ = self.gru(hidden)
        output = self.softmax(self.fc(hidden))
        # print(f"Decoder Output Shape: {output.shape}")

        return output


class AsrProbe(nn.Module):
    def __init__(self, out_size, hidden_size=256):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, hidden):
        output = self.softmax(self.linear(hidden))
        return output

