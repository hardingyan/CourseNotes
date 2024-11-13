import math

import torch
from torch import nn
import torch.nn.functional as F

# Canonical implementation of multi-head self attention.
class SelfAttention(nn.Module):

    def __init__(self, Wk, Wq, Wv, Wunifyheads, emb, heads=8, mask=False, kqnorm=False, scalefactor=None):
        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads

        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([s])
            self.qln = nn.LayerNorm([s])

        self.scalefactor = 1 / math.sqrt(s) if scalefactor is None else scalefactor

        self.KMM = nn.Linear(emb, emb, bias = False)
        self.QMM = nn.Linear(emb, emb, bias = False)
        self.VMM = nn.Linear(emb, emb, bias = False)
        self.unifyheadsMM = nn.Linear(emb, emb, bias = False)

        assert self.KMM.weight.shape == Wk.shape, "Wk Shape mismatch!"
        assert self.QMM.weight.shape == Wq.shape, "Wq Shape mismatch!"
        assert self.VMM.weight.shape == Wv.shape, "Wv Shape mismatch!"
        assert self.unifyheadsMM.weight.shape == Wunifyheads.shape, "Wunifyheads Shape mismatch!"

        self.KMM.weight = nn.Parameter(Wk)
        self.QMM.weight = nn.Parameter(Wq)
        self.VMM.weight = nn.Parameter(Wv)
        self.unifyheadsMM.weight = nn.Parameter(Wunifyheads)

    def forward(self, x):
        
        b, t, e = x.size()
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self})'

        h = self.heads
        s = e // h

        keys = self.KMM(x)
        querys = self.QMM(x)
        values = self.VMM(x)

        keys = keys.view(b, t, h, s)
        querys = querys.view(b, t, h, s)
        values = values.view(b, t, h, s)

        if self.kqnorm: 
            keys = self.kln(keys)
            querys = self.qln(querys)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        querys = querys.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        weight = torch.bmm(querys, keys.transpose(1, 2))
        weight = weight * self.scalefactor

        assert weight.size() == (b * h, t, t)

        if self.mask:
            mask = torch.ones_like(weight).triu(diagonal = 1)
            weight = weight.masked_fill(mask == 1, float('-inf'))

        dot = F.softmax(weight, dim = 2)

        out =  torch.bmm(dot, values).view(b, h, t, s)

        out = out.transpose(1, 2).contiguous().view(b, t, h * s) # h * s = e

        return self.unifyheadsMM(out)