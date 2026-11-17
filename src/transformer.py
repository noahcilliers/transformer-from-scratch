import torch 
import torch.nn as nn
from attn import MultiHeadAttn

class TransformerBlock(nn.Module):
    def __init__(self, dim, head, mask_tf=False):
        """
        x has shape (batch x seq_len x embs)


        """
        super().__init__()
        self.mha = MultiHeadAttn(dim, dim, head, mask_tf)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

    
    def forward(self, x):
        
        x = x + self.mha(self.layer_norm1(x))

        x = x + self.ffn(self.layer_norm2(x))
        
        return x