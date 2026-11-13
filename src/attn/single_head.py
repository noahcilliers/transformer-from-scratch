import torch
import torch.nn as nn
import math

class SingleHeadAttn(nn.Module):
    def __init__(self, embs_dim, dim):
        super().__init__()
        # projection matrices
        self.Q = nn.Linear(embs_dim, dim)
        self.K = nn.Linear(embs_dim, dim)
        self.V = nn.Linear(embs_dim, dim)

        self.dim = dim
        
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

        #softmax rows
        X = torch.softmax(X, dim=-1)

        # get values
        X = X @ V

        return X