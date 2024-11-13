import math

import torch
from torch import nn
import torch.nn.functional as F

def maskHelper(matrices, maskval = 0.0, mask_diagonal = True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval

class SelfAttentionRef(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, Wk, Wq, Wv, Wunifyheads, emb, heads=8, mask=False, kqnorm=False, scalefactor=None):
        """

        :param Wk: shape is (emb, emb)
        :param Wq: shape is (emb, emb)
        :param Wv: shape is (emb, emb)
        :param Wunifyheads: shape is (emb, emb)

        :param emb: The dimension of the input and output vectors.
        :param heads: The number of heads (parallel executions of the self-attention)
        :param mask: Whether to apply an autoregressive mask.
        :param kqnorm: Whether to apply layer normalization to the keys and queries.
        :param scalefactor: Multiplier for the attention weights. If none, the default `1/sqrt(emb/heads)` is used,
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        # - We will break the embedding into `heads` chunks and feed each to a different attention head
        s = emb // heads

        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([s])
            self.qln = nn.LayerNorm([s])

        self.scalefactor = 1/math.sqrt(emb // heads) if scalefactor is None else scalefactor

        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)
        self.unifyheads = nn.Linear(emb, emb, bias=False)

        assert Wk.shape == self.tokeys.weight.shape, "Wk Shape mismatch!"
        assert Wq.shape == self.toqueries.weight.shape, "Wq Shape mismatch!"
        assert Wv.shape == self.tovalues.weight.shape, "Wv Shape mismatch!"
        assert Wunifyheads.shape == self.unifyheads.weight.shape, "Wunifyheads Shape mismatch!"

        self.tokeys.weight = nn.Parameter(Wk)
        self.toqueries.weight = nn.Parameter(Wq)
        self.tovalues.weight = nn.Parameter(Wv)
        self.unifyheads.weight = nn.Parameter(Wunifyheads)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)

        # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
        #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot * self.scalefactor

        assert dot.size() == (b * h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            maskHelper(dot, maskval = float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # -- dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

        return self.unifyheads(out)