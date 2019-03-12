import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

class FullyConnected(nn.Module):
    def __init__(self, args, fully_connected_shape): 
        self.lin = nn.Linear(fully_connected_shape, self.args.hidden_dim[0])
        self.fc = []

        for i in range(1, len(args.hidden_dim)):
            lin = nn.Linear(self.args.hidden_dim[i-1], self.args.hidden_dim[i])
            self.add_module('layer_' + str(args.num_layers+i) + '_fc_'+str(args.hidden_dim[i]), lin)
            self.fc.append(lin)

        self.dropout = nn.Dropout(args.dropout)
        self.hidden = nn.Linear(self.args.hidden_dim[-1], self.args.num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden = self.lin(x)
        for i in range(len(self.fc)):
            hidden = self.fc[i](hidden)
        hidden = self.dropout(hidden)
        logit = self.hidden(hidden)

        return logit, hidden
