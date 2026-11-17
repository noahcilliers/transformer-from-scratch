import torch.nn as nn
from transformer_block.py import TransformerBlock
"""
so lets try to think what a GNN class really looks like

like we just run attn on all the connected nodes, and then add them all up and then output

"""

class GNN(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.transformer = nn.Sequential(
            TransformerBlock(dim, heads)
            TransformerBlock(dim, heads)
            TransformerBlock(dim, heads)
        )
        self.linear_in = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim, dim)

    def forward(self, x, G):
        """
        linear on x

        MHA on all g in G

        mean of all g

        g + x

        linear out

        """