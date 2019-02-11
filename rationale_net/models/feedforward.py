
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb


class FeedForward(nn.Module):
    def __init__(self, args, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.args = args

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(self.args.dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
        
