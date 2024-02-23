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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from models.lstm_models import *

def one_hot(labels, label_dict):
    result = np.zeros((len(labels), len(label_dict) + 1))
    for i, ell in enumerate(labels):
        result[i, int(ell.item())] = 1

    return result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--batch-size')
    parser.add_argument('--hidden-size')
    parser.add_argument('--layers')
    parser.add_argument('--label-set')
    parser.add_argument('--label-scp')
    parser.add_argument('--val-label-scp')
    parser.add_argument('--feat-scp')
    parser.add_argument('--val-feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--val-mean-var')
    parser.add_argument('--apc-param')
    parser.add_argument('--pred-param')
    parser.add_argument('--pred-param-output')
    parser.add_argument('--learning-rate')
    parser.add_argument('--stoch-prob', type=float)
    parser.add_argument('--linear-decay', type=bool)
    parser.add_argument('--init', action="store_true")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--target-len', type=int)
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
    data, labels, feat_lengths = zip(*batch)

    # Pad features and labels
    data = pad_sequence(data, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

    return data, labels, feat_lengths

print(' '.join(sys.argv))
args = parse_args()

# INDEX USED FOR PADDING
padding_index = 0

label_dict = load_label_dict(args.label_set)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

print('device: ', device)
print()

pretrained_lstm = UniLstm(40, args.hidden_size, args.layers)

if args.type == "skip-baseline":
    pretrained_lstm = UniLstmSkip(40, args.hidden_size, args.layers)

if args.type == "stochastic":
    pretrained_lstm = UniLstmStochastic(40, args.hidden_size, args.layers, args.stoch_prob, args.linear_decay)

model = ProbeLSTM(pretrained_lstm, args.hidden_size, len(label_dict) + 1)
model.to(device)

loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_index)
total_param = list(model.fc.parameters())

opt = torch.optim.Adam(total_param, lr=args.learning_rate)

step_size = args.learning_rate

if args.init:
    torch.save(
        {
            'fc': model.fc.state_dict(),
            'opt': opt.state_dict()
        },
        args.pred_param_output)
    exit()

if args.apc_param:
    checkpoint = torch.load(args.apc_param)
    pretrained_lstm.load_state_dict(checkpoint['model'])

    # Freeze the weights of the pretrained LSTM
    for param in pretrained_lstm.parameters():
        param.requires_grad = False

if args.pred_param:
    checkpoint = torch.load(args.pred_param)
    model.fc.load_state_dict(checkpoint['fc'])
    opt.load_state_dict(checkpoint['opt'])

model.lstm.requires_grad_(False)

rand_eng = Random(args.seed)

feat_mean, feat_var = load_mean_var(args.feat_mean_var)
val_mean, val_var = load_mean_var(args.val_mean_var)

dataset = WsjDataset(args.feat_scp, args.label_scp, label_dict, feat_mean, feat_var, shuffling=True, rand=rand_eng)
val_dataset = WsjDataset(args.val_feat_scp, args.val_label_scp,  label_dict, val_mean, val_var, shuffling=False)
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
start_time = time.time()
model.fc.train()

for batch_idx, (feat, labels, feat_lengths) in enumerate(dataloader):
    opt.zero_grad()

    feat_lengths = torch.Tensor(feat_lengths)
    _, indices = torch.sort(feat_lengths, descending=True)

    feat = Variable(feat[indices]).to(device)
    labels = Variable(labels[indices]).to(device)
    feat_lengths = Variable(feat_lengths[indices]).tolist()

    # Convert labels to one-hot representation
    labels = labels.to(torch.long)

    # Mask indicating the padded positionsp
    mask = (labels != padding_index).float()

    hidden, pred = model(feat, feat_lengths)  # Forward pass
    _, nframes, nclass = pred.shape

    loss = loss_fn(pred.permute(0, 2, 1), labels)

    print('sample: ', batch_idx)
    print('loss: {:.6}'.format(loss.item()))

    loss.backward()  # Backward pass

    # Print gradient norm
    parameters = torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])
    grad_norm = torch.norm(parameters, p=2)
    print('grad norm: {:.6}'.format(grad_norm))

    # Update parameters
    opt.step()

    print()

end_time = time.time()
print(f"epoch time: {end_time - start_time}")

print("---------------------------\n")

# Validate the model
model.eval()
test_loss = 0
total_accuracy = 0
with torch.no_grad():
    for idx, (feat, labels, feat_lengths) in enumerate(val_dataloader):
        feat_lengths = torch.Tensor(feat_lengths)
        feat = feat.to(device)
        hidden, pred = model(feat, feat_lengths.tolist())
        labels = labels.to(torch.long)
   
        pred = pred.to(device)
        labels = labels.to(device)

        # Convert labels to one-hot representation
        loss = loss_fn(pred.permute(0, 2, 1), labels)
        test_loss += loss.item()

        # Determine accuracy
        labels = labels.squeeze(0)
        target_labels = torch.tensor(one_hot(labels, label_dict)).to(device)
        predicted_labels = torch.argmax(pred, dim=2).squeeze(0)
        cur_accuracy = torch.sum(predicted_labels == labels).item() / len(labels)

        total_accuracy += cur_accuracy

        print(f"Validated {idx}/{len(val_dataloader)}...")
        print(f"Current Accuracy: {cur_accuracy}")

print()
print(f"Average Validation Loss: {test_loss / len(val_dataloader)}")
print(f"Average Accuracy: {total_accuracy / len(val_dataloader)}")

torch.save(
    {
        'fc': model.fc.state_dict(),
        'opt': opt.state_dict()
    },
    args.pred_param_output)

