import torch
import torch.nn as nn
from .single_head import SingleHeadAttn

"""
so this class does multi head attention

owning n attention heads it does attention for all of them
then concats them together and puts them through a linear out

"""

class MultiHeadAttn(nn.Module):
    def __init__(self, embs_dim, dim, heads, mask_tf=False):
        super().__init__()
        assert dim % heads == 0
        self.attn_heads = nn.ModuleList(SingleHeadAttn(embs_dim, dim//heads, mask_tf) for i in range(heads))

        self.out = nn.Linear(dim, dim)
        self.dim = dim
        self.heads = heads

    def forward(self, X):
        # collect outs as a list and then concat together
        outs = [head(X) for head in self.attn_heads]
        return self.out(torch.cat(outs, dim=-1))