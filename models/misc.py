import torch as th
import torch.nn as nn

# post-LN
class ResidualLayer(nn.Module):
    def __init__(
        self, in_dim = 100,
        out_dim = 100,
        dropout = 0.,
    ):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = self.dropout(x)
        return self.layer_norm(self.dropout(self.lin2(x).relu()) + x)

class LogDropout(nn.Module):
    def __init__(
        self, p,
    ):
        super(LogDropout, self).__init__()
        self.p = p

    def forward(self, x, column_dropout=False):
        if self.training and self.p > 0:
            if column_dropout:
                annihilate_mask = th.empty_like(x).fill_(self.p).bernoulli().bool()
            else:
                annihilate_mask = (th.empty(x.shape[-1])
                    .fill_(self.p).bernoulli().bool()[None].expand(x.shape)
                )
            return x.masked_fill(annihilate_mask, float("-inf"))
        else:
            return x

class LogDropoutM(nn.Module):
    def __init__(
        self, p,
    ):
        super(LogDropoutM, self).__init__()
        self.p = p

    def forward(self, x, annihilate_mask=None):
        if self.training and self.p > 0 and annihilate_mask is None:
            return x
        elif self.training and self.p > 0 and annihilate_mask is not None:
            while annihilate_mask.dim() < x.dim():
                annihilate_mask = annihilate_mask.unsqueeze(0)
            annihilate_mask = annihilate_mask.expand(x.shape)
            return x.masked_fill(annihilate_mask, float("-inf"))
        else:
            return x


