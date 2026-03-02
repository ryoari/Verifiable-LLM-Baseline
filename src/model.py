# linear model as the starting point, nanoGPT next
# Unblock this code block to check the linear model out
# :ATTENTION: Remember to block out the remaining code blocks
 
'''
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        x = x.mean(dim=0)
        logits = self.linear(x)
        return logits

'''

# Determinitic NanoGPT model
# Andrej Karpathy's NanoGPT, Source:https://github.com/karpathy/nanoGPT

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = num_heads
        self.n_embd = embed_dim

        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, num_heads=2, max_seq_len=32):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, embed_dim),
            wpe = nn.Embedding(max_seq_len, embed_dim),
            h = nn.ModuleList([Block(embed_dim, num_heads, max_seq_len)]),
            ln_f = nn.LayerNorm(embed_dim)
        ))
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb +pos_emb

        for block in self.transformer.h:
            x= block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
