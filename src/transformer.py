import torch 
import torch.nn as nn
from .multi_head import MultiHeadAttn

class TransformerBlock(nn.Module):
    def __init__(self, embs, dim, head):
        """
        x has shape (batch x seq_len x embs)


        """
        super().__init__()
        self.mha = MultiHeadAttn(embs, dim, head)
        self.layer_norm = nn.LayerNorm(embs)

        self.upsample = nn.Sequential(
            nn.Linear(embs, 4 * dim)
            nn.ReLU()
        )

        self.down = nn.Sequential(
            nn.Linear(4 * dim, dim)
            nn.ReLU()
        )


    def forward(self, x):
        
        x_hat = self.mha(x)
  
        x = x + x_hat
        y = self.layer_norm(x)

        y = self.upsample(y)
        y = self.down(y)
        x = x + y
        
        return self.layer_norm(x)