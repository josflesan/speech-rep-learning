#!/usr/bin/env python3

import sys
import math
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import argparse
import json
import time
import numpy as np
from rand import *
from dataprep import *
from data import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

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

    def forward(self, x):
        # List to store intermediate outputs for skip connections
        skip_connections = []

        # Forward pass through LSTM layers with skip connections and stochastic depth
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i == 0:
                out, _ = lstm_layer(x)
            else:
                # Determine probability for layer
                drop_prob = 1 - (i / len(self.lstm_layers)) * (1 - self.stochastic_depth_prob) if self.linear else self.stochastic_depth_prob

                # Add packed sequences together
                unpacked_out, lengths_out = pad_packed_sequence(out, batch_first=True)
                unpacked_skip, lengths_skip = pad_packed_sequence(skip_connections[-1], batch_first=True)
                added_packed_sequence = pack_padded_sequence(unpacked_out + unpacked_skip, lengths_out, batch_first=True)

                # Apply stochastic depth to skip connections
                if self.training and torch.rand(1).item() < drop_prob:
                    out = added_packed_sequence
                else:
                    out, _ = lstm_layer(added_packed_sequence)

            skip_connections.append(out)

        # Unpack the PackedSequence
        out, _ = pad_packed_sequence(out, batch_first=True)

        return out, self.proj(out)

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

    def forward(self, x):
        # List to store intermediate outputs for skip connections
        skip_connections = []

        # Forward pass through LSTM layers with skip connections
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i == 0:
                out, _ = lstm_layer(x)
            else:
                unpacked_out, lengths_out = pad_packed_sequence(out, batch_first=True)
                unpacked_skip, lengths_skip = pad_packed_sequence(skip_connections[-1], batch_first=True)
                added_packed_sequence = pack_padded_sequence(unpacked_out + unpacked_skip, lengths_out, batch_first=True)
                out, _ = lstm_layer(added_packed_sequence)

            skip_connections.append(out)

        # Unpack the PackedSequence
        out, _ = pad_packed_sequence(out, batch_first=True)
        
        return out, self.proj(out)

class UniLstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout,
            bidirectional=False, batch_first=True)
        self.proj = torch.nn.Linear(hidden_size, input_size)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)

        # Unpack the PackedSequence
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)

        return hidden, self.proj(hidden)

class ProbeLSTM(nn.Module):
    def __init__(self, pretrained_lstm, acoustic_features, num_classes):
        super().__init__()
        self.lstm = pretrained_lstm
        self.fc = nn.Linear(acoustic_features, num_classes)

    def forward(self, x):
        # Forward pass through the LSTM
        lstm_out, lstm_proj = self.lstm(x)

        # Project the output using a linear layer
        output = self.fc(lstm_proj)

        # Log-softmax activation
        output = F.log_softmax(output, dim=1)

        return output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--batch-size')
    parser.add_argument('--hidden-size')
    parser.add_argument('--layers')
    parser.add_argument('--label-set')
    parser.add_argument('--label-scp')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--apc-param')
    parser.add_argument('--pred-param')
    parser.add_argument('--pred-param-output')
    parser.add_argument('--learning-rate')
    parser.add_argument('--init')
    parser.add_argument('--stoch-prob', type=float)
    parser.add_argument('--linear-decay', type=bool)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--type', choices=["normal", "skip-baseline", "stochastic"], type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)

    args = parser.parse_args()

    if args.config:
        f = open(args.config)
        config = json.load(f)
        f.close()

        for k, v in config.items():
            if k not in args.__dict__ or args.__dict__[k] is None:
                args.__dict__[k] = v
        
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print()

    return args

def collate_fn(batch):
    # Separate data and labels
    data, labels = zip(*batch)

    # Sort the batch by sequence length in decreasing order
    data = sorted(data, key=lambda x: x.shape[0], reverse=True)
    feat_lengths = [seq.shape[0] for seq in data]

    # Get the maximum sequence length in the batch
    max_len = data[0].shape[0]

    # Pad sequences to the maximum length
    padded_data = [torch.cat([seq, torch.zeros(max_len - seq.shape[0], seq.shape[1])], dim=0) for seq in data]

    # Convert the padded sequences to a tensor
    data_tensor = torch.stack(padded_data)

    # Create a tensor of sequence lengths
    lengths = [seq.shape[0] for seq in padded_data]

    # Pack the padded sequence
    packed_sequence = pack_padded_sequence(data_tensor, lengths, batch_first=True, enforce_sorted=False)
    
    # Pad labels
    labels_lengths = [seq.shape[0] for seq in labels]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    return packed_sequence, labels_padded, feat_lengths, labels_lengths

print(' '.join(sys.argv))
args = parse_args()

label_dict = load_label_dict(args.label_set)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

print('device: ', device)
print()

pretrained_lstm = UniLstm(40, args.hidden_size, args.layers)
model = ProbeLSTM(pretrained_lstm, 40, len(label_dict) + 1)
model = model.to(torch.float)

if args.type == "skip-baseline":
    pretrained_lstm = UniLstmSkip(40, args.hidden_size, args.layers)

if args.type == "stochastic":
    pretrained_lstm = UniLstmStochastic(40, args.hidden_size, args.layers, args.stoch_prob, args.linear_decay)

# ----------------------------------------------------------------- #
# Infinite losses mainly occur when the inputs are too short to be  #
# aligned to the targets                                            #
# ----------------------------------------------------------------- #
loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
total_param = list(model.fc.parameters())

opt = torch.optim.Adam(total_param, lr=args.learning_rate)

step_size = args.learning_rate
grad_clip = args.grad_clip

if args.apc_param:
    checkpoint = torch.load(args.apc_param, map_location=device)
    pretrained_lstm.load_state_dict(checkpoint['model'])

    # Freeze the weights of the pretrained LSTM
    for param in pretrained_lstm.parameters():
        param.requires_grad = False

if args.init:
    torch.save(
        {
            'fc': model.fc.state_dict(),
            'opt': opt.state_dict()
        },
        args.pred_param_output)
    exit()

model.to(device)

rand_eng = Random(args.seed)

feat_mean, feat_var = load_mean_var(args.feat_mean_var)
dataset = WsjDataset(args.feat_scp, args.label_scp, label_dict, feat_mean, feat_var, shuffling=True, rand=rand_eng)
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
start_time = time.time()

for batch_idx, (packed_feat, labels, feat_lengths, label_lengths) in enumerate(dataloader):
    packed_feat = packed_feat.to(device)
    opt.zero_grad()

    pred = model(packed_feat)  # Forward pass

    # Unpack the sequence
    padded_feat, lengths = pad_packed_sequence(packed_feat, batch_first=True)
    loss = loss_fn(pred.permute(1, 0, 2), labels, feat_lengths, label_lengths)

    print('sample: ', batch_idx)
    print('loss: {:.6}'.format(loss.item()))

    loss.backward()  # Backward pass

    # Compute gradient norm
    total_norm = 0
    for p in total_param:
        if p.grad is None:
            continue

        n = p.grad.norm(2).item()
        total_norm += n * n
    grad_norm = math.sqrt(total_norm)

    print('grad norm: {:.6}'.format(grad_norm))

    # Update learning rate
    param_0 = total_param[0][0, 0].item()

    # If gradient norm is too large, scale down learning rate
    if grad_norm > grad_clip:
        opt.param_groups[0]['lr'] = step_size / grad_norm * grad_clip
    else:
        opt.param_groups[0]['lr'] = step_size

    # Update parameters
    opt.step()

    # Print learning rate and parameter update
    param_0_new = total_param[0][0, 0].item()

    print('param: {:.6}, update: {:.6}, rate: {:.6}'.format(param_0, param_0_new - param_0, (param_0_new - param_0) / param_0))

    print()

end_time = time.time()
print(f"epoch time: {end_time - start_time}")

torch.save(
    {
        'fc': model.fc.state_dict(),
        'opt': opt.state_dict()
    },
    args.pred_param_output)

