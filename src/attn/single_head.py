import torch
import torch.nn as nn
import math

class SingleHeadAttn(nn.Module):
    def __init__(self, embs_dim, dim, mask_tf=False):
        super().__init__()
        # projection matrices
        self.Q = nn.Linear(embs_dim, dim)
        self.K = nn.Linear(embs_dim, dim)
        self.V = nn.Linear(embs_dim, dim)

        self.dim = dim
        self.embs_dim = embs_dim
        self.mask_tf = mask_tf
    def forward(self, X):
        # here x represents our inputs 
        #   x_1(5, 5, 5)
        #   x_2(5, 7, 4)
        #   x_3(6. 7, 9)
        # so these are all different words in shape n(words) x d (emb_dim)
        # first we need to project x into our proper spaces
        Q = self.Q(X) #n x d
        K = self.K(X)
        V = self.V(X)

        # transpose the second and third dim
        # since we have batch x n x d, we swap the d and n
        # X is now n x n
        X = Q @ K.transpose(-2, -1)

        #scale
        X = X * (1/math.sqrt(self.dim))

        # here is where we add our mask for decoding
        # this makes it so words dont attend to future positions
        # mask is upper triange
        #   (0, -inf, -inf)
        #   (0,    0, -inf)
        #   (0,    0,    0)
        # is n x n where n is the number of words
        if self.mask_tf:
            n = X.size(-1)
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=X.device), diagonal=1)
            X = X.masked_fill(mask, float('-inf'))


        #softmax rows
        X = torch.softmax(X, dim=-1)

        # get values
        X = X @ V

        return X