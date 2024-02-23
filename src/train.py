#!/usr/bin/env python3

import sys
import math
import data
import torch.nn as nn
import torch.optim
import argparse
import json
import time
import numpy as np
from rand import *
from torch.autograd import Variable
from dataprep import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.transformer_models import *
from models.lstm_models import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--feat-scp')
    parser.add_argument('--feat-mean-var')
    parser.add_argument('--val-feat-scp')
    parser.add_argument('--val-feat-mean-var')
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--step-size', type=float)
    parser.add_argument('--clip-norm', type=float)
    parser.add_argument('--time-shift', type=int)
    parser.add_argument('--init', action="store_true")
    parser.add_argument('--param')
    parser.add_argument('--param-output')
    parser.add_argument('--stoch-prob', type=float)
    parser.add_argument('--linear-decay', type=bool)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--type', choices=["normal", "skip-baseline", "stochastic", "transformer", "transformer-stoch", "fractal-lstm"], type=str)
    
    # Transformer-Specific Arguments
    parser.add_argument('--nhead', type=int)

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
        print('{}: {}'.format(k, v))
    print()

    if args.clip_norm == 0.0:
        args.clip_norm = float('inf')

    return args

def collate_fn(batch):
    seq, lengths = zip(*batch)

    seq = pad_sequence(seq, batch_first=True)
    lengths = torch.Tensor(lengths)

    return seq, lengths

print(' '.join(sys.argv))
args = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device:', device)
print()

isTransformer = False
model = UniLstm(40, args.hidden_size, args.layers)

if args.type == "skip-baseline":
    model = UniLstmSkip(40, args.hidden_size, args.layers)

if args.type == "stochastic":
    model = UniLstmStochastic(40, args.hidden_size, args.layers, args.stoch_prob, args.linear_decay)

if args.type == "transformer":
    isTransformer = True
    model = ApcTransformerEncoder(40, args.hidden_size, args.layers, args.nhead, dropout=0.0, device=device)

if args.type == "transformer-stoch":
    isTransformer = True
    model = ApcStochasticTransformerEncoder(40, args.hidden_size, args.layers, args.nhead, args.stoch_prob, args.linear_decay, device=device)

if args.type == "fractal-lstm":
    model = FractalLSTM(40, args.hidden_size, 2, args.layers, device=device)

loss_fn = torch.nn.MSELoss(reduction='sum')
total_param = list(model.parameters())

opt = torch.optim.Adam(total_param, lr=args.step_size)
model.to(device)

if args.param:
    checkpoint = torch.load(args.param)
    model.load_state_dict(checkpoint['model'])
    opt.load_state_dict(checkpoint['opt'])

if args.init:
    torch.save(
        {
            'model': model.state_dict(),
            'opt': opt.state_dict()
        },
        args.param_output)
    exit()

step_size = args.step_size
shift = args.time_shift

feat_mean, feat_var = load_mean_var(args.feat_mean_var)
val_feat_mean, val_feat_var = load_mean_var(args.val_feat_mean_var)

rand_eng = Random(args.seed)

librispeechdataset = data.LibriSpeechDataset(args.feat_scp, feat_mean, feat_var, shuffling=True, rand=rand_eng)
devlibrispeechdataset = data.LibriSpeechDataset(args.val_feat_scp, val_feat_mean, val_feat_var, shuffling=True, rand=rand_eng)
batched_dataloader = DataLoader(librispeechdataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(devlibrispeechdataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
start_time = time.time()
model.train()

padding_val = 0
brokenPipeline = False  # Flag to detect NaN loss

for batch_idx, (seq, lengths) in enumerate(batched_dataloader):      
    seq = seq.to(device)
    opt.zero_grad()  # Clear gradient

    _, indices = torch.sort(lengths, descending=True)

    seq = Variable(seq[indices])
    lengths = Variable(lengths[indices])
    
    # If transformer, no need to pass in lengths to model
    pred = None
    if isTransformer:
        padding_mask = (seq[:, :, 0] == padding_val)
        pred, _ = model(seq, padding_mask)
    else:
        pred, _ = model(seq, lengths) # Forward pass

    # Create a mask
    indices = torch.arange(seq.size(1))[None, :]
    mask = (indices < lengths[:, None])[:, shift:]
    non_zero_elements = mask.sum()
    mask = mask.to(device)

    # Compute loss and average over non-padded elements
    # loss = (loss_fn(pred[:, :-shift, :], seq[:, shift:, :]) * mask.unsqueeze(-1).float()).sum()
    # loss = loss / non_zero_elements
    loss = loss_fn(pred[:, :-shift, :], seq[:, shift:, :])

    # Check if loss is NaN
    if torch.isnan(loss):
        print(f"\nNaN loss was encountered")
        brokenPipeline = True
        break

    print('sample:', batch_idx)
    print('loss: {:.6}'.format(loss.item() / ((seq.shape[1] - shift) * args.batch_size)))

    loss.backward() # Backward pass

    # Clip the gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

    parameters = torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])
    grad_norm = torch.norm(parameters, p=2)
    print('grad norm: {:.6}'.format(grad_norm))

    # Update parameters
    opt.step()

    print()

end_time = time.time()
print(f"epoch time: {end_time - start_time}")

######################
##### Validation #####
######################

model.eval()
val_losses = []
with torch.set_grad_enabled(False):
    for (val_seq, val_lengths) in val_dataloader:
        val_seq = val_seq.to(device)
        
        _, indices = torch.sort(val_lengths, descending=True)
        val_seq = Variable(val_seq[indices])
        val_lengths = Variable(val_lengths[indices])
        
        if isTransformer:
            padding_mask = (seq[:, :, 0] == padding_val)
            pred, _ = model(val_seq, padding_mask)  # No padding mask since no batching
        else:
            pred, _ = model(val_seq, val_lengths.tolist())

        # Compute loss
        loss = loss_fn(pred[:, :-shift, :], val_seq[:, shift:, :])
        val_losses.append(loss.item() / ((val_seq.size(1) - shift) * args.batch_size))


print("----------- VALIDATION ------------")
print("mean validation loss: {:.6}".format(np.mean(val_losses)))


# Save model
torch.save(
    {
        'model': model.state_dict(),
        'opt': opt.state_dict()
    },
    args.param_output if not brokenPipeline else 'param-broken')

