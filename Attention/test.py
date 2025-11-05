import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from TransformerRef import SelfAttentionRef
from Transformer import SelfAttention

def diff(act, ref):
    act = act.detach().numpy()
    ref = ref.detach().numpy()

    assert act.shape == ref.shape

    ref = ref.flatten()
    act = act.flatten()

    absDiff = abs(ref - act)
    maxAbsDiff = max(abs(ref - act))
    rangeV = max(abs(ref))
    relDiff = maxAbsDiff / rangeV

    print('[Absolute] avgErr= %g, maxErr= %g' % (np.average(maxAbsDiff), maxAbsDiff))
    print('[Relative] avgErr= %g%%, maxErr=%g%%' % (np.average(relDiff) * 100, relDiff.max() * 100))

    None

if __name__ == "__main__":
    emb = 64

    WShape = (emb, emb)

    Wq = torch.rand(WShape, dtype = torch.float)
    Wk = torch.rand(WShape, dtype = torch.float)
    Wv = torch.rand(WShape, dtype = torch.float)
    Wunifyheads = torch.rand(WShape, dtype = torch.float)

    attentionRef = SelfAttentionRef(Wq, Wk, Wv, Wunifyheads, emb, heads=4, mask=False, kqnorm=False, scalefactor=None)
    attentionAct = SelfAttention(Wq, Wk, Wv, Wunifyheads, emb, heads=4, mask=False, kqnorm=False, scalefactor=None)

    b = 1
    t = 10

    input = torch.rand((b, t, emb), dtype = torch.float)

    outputRef = attentionRef(input)
    outputAct = attentionAct(input)

    diff(outputAct, outputRef)
