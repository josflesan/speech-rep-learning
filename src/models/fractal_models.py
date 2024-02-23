import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

### COMPONENTS ###

class Head(nn.Module):
    def __init__(self, n_embed, head_size, dropout=0.0, block_size=32, device='cpu'):
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).to(device))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fil(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, head_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(x))
        return out
    
class FractalBlock(nn.Module):
    def __init__(self, n_embed, n_head, n_cols, dropout=0.0):
        super().__init__()
        self.n_cols = n_cols
        self.dropout = nn.Dropout(dropout)
        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(n_cols)])
        self.max_depth = 2 ** (n_cols - 1)

        dist = self.max_depth
        self.count = [0] * self.max_depth

        for col in self.columns:
            for i in range(self.max_depth):
                if (i + 1) % dist == 0:
                    module = MultiHeadAttention(n_embed, n_head, n_embed//n_head)
                    self.count[i] += 1
                else:
                    module = None
                
                col.append(module)
            
            dist //= 2

    def forward(self, x):
        out = [x for _ in range(self.n_cols)]
        for i in range(self.max_depth):
            st = self.n_cols - self.count[i]
            cur_outs = []
            for c in range(st, self.n_cols):
                cur_in = out[c]
                cur_module = self.columns[c][i]
                cur_outs.append(cur_module(cur_in))

            n_out = torch.stack(cur_outs)
            n_out = n_out.mean(dim=0)  # Average over all columns

            for c in range(st, self.n_cols):
                out[c] = n_out
        
        return self.dropout(out[-1])
    
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head, n_cols):
        super().__init__()
        self.sa_head = FractalBlock(n_embed, n_head, n_cols)
        self.ffw = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x
    
class FractalTransformer(nn.Module):
    def __init__(self, input_size, n_embed, block_size, n_head, n_cols, n_layer, device='cpu'):
        super().__init__()

        self.device = device

        self.token_embedding_table = nn.Embedding(input_size, n_embed).to(device)
        self.position_embedding_table = nn.Embedding(block_size, n_embed).to(device)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, n_cols=n_cols) for _ in range(n_layer)])
        self.proj = nn.Linear(n_embed, input_size)

    def forward(self, idx):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx).to(self.device)
        pos_emb = self.position_embedding_table(torch.arange(T).to(self.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        predicted_mel = self.proj(x)

        return predicted_mel

### PRETRAIN MODELS ###


### PROBE MODELS ###

