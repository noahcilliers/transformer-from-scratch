import torch
import torch.nn
from single_head import SingleHeadAttn

class MultiHeadAttn(nn.Module):
    def __init__(self, embs_dim, dim):
        super().__init__()
        #lets just make 8 heads
        self.head1 = SingleHeadAttn(embs_dim, dim/8)
        self.head2 = SingleHeadAttn(embs_dim, dim/8)
        self.head3 = SingleHeadAttn(embs_dim, dim/8)
        self.head4 = SingleHeadAttn(embs_dim, dim/8)
        self.head5 = SingleHeadAttn(embs_dim, dim/8)
        self.head6 = SingleHeadAttn(embs_dim, dim/8)
        self.head7 = SingleHeadAttn(embs_dim, dim/8)
        self.head8 = SingleHeadAttn(embs_dim, dim/8)

        self.out = nn.Linear(embs_dim, dim)

    def forward(self, X):

        x_1 = self.head1(X)
        x_2 = self.head2(X)
        x_3 = self.head3(X)
        x_4 = self.head4(X)
        x_5 = self.head5(X)
        x_6 = self.head6(X)
        x_7 = self.head7(X)
        x_8 = self.head8(X)

        X = torch.concat((x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8), dim=0)


        return self.out(X)