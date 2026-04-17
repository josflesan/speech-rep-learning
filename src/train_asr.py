#!/usr/bin/env python3

import sys
import math
import time
import torch.nn as nn
import torch.optim
import argparse
import json
import numpy as np
from rand import *
from dataprep import *
from data import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from models.lstm_models import *
from models.asr_models import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--label-set')
    parser.add_argument('--label-scp')
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--apc-param')
    parser.add_argument('--asr-param')
    parser.add_argument('--pred-param-output')
    parser.add_argument('--seed', type=int)

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

    if args.dropout is None:
        args.dropout = 0.0

    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print()

    return args

def collate_fn(batch):
    seq, labels, feat_lengths, label_length = zip(*batch)

    seq = pad_sequence(seq, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    feat_lengths = torch.Tensor(feat_lengths)
    label_length = torch.Tensor(label_length)

    return seq, labels, feat_lengths, label_length

print(' '.join(sys.argv))
args = parse_args()

label_dict = load_label_dict(args.label_set)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'device: {device}')
print()

pretrained_net = UniLstm(40, args.hidden_size, args.layers)
# encoder = AsrEncoder()
# decoder = AsrDecoder(len(label_dict))
pretrained_net.to(device)
# encoder.to(device)
# decoder.to(device)
probe = AsrProbe(len(label_dict) + 1)

loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
total_param = list(probe.parameters())

opt = torch.optim.Adam(total_param, lr=args.learning_rate)
probe.to(device)

if args.asr_param:
    ckpt = torch.load(args.asr_param)
    # encoder.load_state_dict(ckpt['enc'])
    # decoder.load_state_dict(ckpt['dec'])
    probe.load_state_dict(ckpt['probe'])
    opt.load_state_dict(ckpt['opt'])

if args.apc_param:
    ckpt = torch.load(args.apc_param)
    pretrained_net.load_state_dict(ckpt['model'])

if args.init:
    torch.save(
        {
            'probe': probe.state_dict(),
            'opt': opt.state_dict()
        },
        args.pred_param_output)
    exit()

# encoder.to(device)
# decoder.to(device)
pretrained_net.requires_grad_(False)

step_size = args.learning_rate

feat_mean, feat_var = load_mean_var(args.feat_mean_var)

rand_eng = Random(args.seed)

dataset = WsjDataset(args.feat_scp, args.label_scp, label_dict, feat_mean, feat_var, shuffling=True, rand=rand_eng)
dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, drop_last=True)
start_time = time.time()
# encoder.train()
# decoder.train()
probe.train()

brokenPipeline = False  # Flag to detect NaN loss

train_losses = []
for batch_idx, (feat, labels, feat_lengths, label_lengths) in enumerate(dataloader):
    opt.zero_grad()

    feat_lengths = torch.Tensor(feat_lengths)
    _, indices = torch.sort(feat_lengths, descending=True)

    feat = Variable(feat[indices]).to(device)
    labels = Variable(labels[indices]).to(device)
    label_lengths = Variable(label_lengths[indices]).to(device)
    feat_lengths = Variable(feat_lengths[indices]).tolist()

    # Get hidden representations from pretrained model
    _, hidden = pretrained_net(feat, feat_lengths)
    # encoder_out = encoder(hidden)
    # decoder_out = decoder(encoder_out)
    # decoder_out = decoder_out.permute(1, 0, 2)
    
    probe_out = probe(hidden)
    probe_out = probe_out.permute(1, 0, 2)

    input_lengths = torch.tensor(feat_lengths, dtype=torch.long)
    label_lengths = label_lengths.to(torch.long)
    
    loss = loss_fn(probe_out, labels, input_lengths, label_lengths)
    train_losses.append(loss)

    if torch.isnan(loss):
        print("Loss is NaN!")
        brokenPipeline = True
        break
    
    print(f'batch: ', batch_idx)
    print(f'loss: {loss.item():.6}')

    loss.backward()

    opt.step()

    print()

print(f"\nEpoch Training Loss Mean: {sum(train_losses) / len(dataloader)}")

torch.save(
    {
        'probe': probe.state_dict(),
        'opt': opt.state_dict()
    },
    args.pred_param_output if not brokenPipeline else "param-broken"
)

