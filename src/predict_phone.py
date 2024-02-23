#!/usr/bin/env python3

import sys
import math
import torch.nn as nn
import argparse
import json
import numpy as np 
from data import WsjDataset
from dataprep import *

from models.lstm_models import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--label-set')
    parser.add_argument('--label-scp')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--apc-param')
    parser.add_argument('--pred-param')
    parser.add_argument('--stoch-prob', type=float)
    parser.add_argument('--linear-decay', type=bool)
    
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
        print(f'{k}: {v}')
    print()

    return args

print(' '.join(sys.argv))
args = parse_args()

label_dict = load_label_dict(args.label_set)

f = open(args.label_set)
id_label = []
for line in f:
    id_label.append(line.strip())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device: ', device)
print()

pretrained_lstm = UniLstm(40, args.hidden_size, args.layers)

if args.type == "skip-baseline":
    pretrained_lstm = UniLstmSkip(40, args.hidden_size, args.layers)

if args.type == "stochastic":
    pretrained_lstm = UniLstmStochastic(40, args.hidden_size, args.layers, args.stoch_prob, args.linear_decay)

model = ProbeLSTM(pretrained_lstm, args.hidden_size, len(id_label) + 1)

ckpt = torch.load(args.pred_param)
model.fc.load_state_dict(ckpt['fc'])

ckpt = torch.load(args.apc_param)
model.lstm.load_state_dict(ckpt['model'])

model.to(device)
model.eval()

feat_mean, feat_var = load_mean_var(args.feat_mean_var)
dataset = WsjDataset(args.feat_scp, args.label_scp, label_dict, feat_mean, feat_var)

total_accuracy = 0
num_substitutions = 0
total_phones = 0
for sample, (feat, target, feat_length) in enumerate(dataset):
    feat = torch.Tensor(feat).to(device)

    nframes, ndim = feat.shape
    feat = feat.reshape(1, nframes, ndim)

    hidden, pred = model(feat, [feat_length])

    _, nframes, nclass = pred.shape
    pred = pred.reshape(nframes, nclass)

    labels = torch.argmax(pred, dim=1)

    result = [id_label[int(e) - 1] for e in labels]
    target_labels = [id_label[int(e) - 1] for e in target]

    cur_accuracy = np.sum(np.array(result) == np.array(target_labels)) / len(labels)
    total_accuracy += cur_accuracy

    print("Predicted: " + ' '.join(result))
    print('.')
    print("Actual: " + ' '.join(target_labels))

    # Accumulate substitutions
    num_substitutions += np.sum(np.array(result) != np.array(target_labels))
    total_phones += len(target_labels)

# Calculate phone error rate (NO SUBSTITUTIONS OR DELETIONS SINCE NO ALIGNMENT)
phone_error_rate = num_substitutions / total_phones

print()
print(f"Average Accuracy: {total_accuracy / len(dataset)}")
print(f"PER: {phone_error_rate}")
