from abc import ABCMeta, abstractmethod, abstractproperty
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from rationale_net.models.fully_connected import FullyConnected
from rationale_net.models.factory import RegisterModel

class AbstractCompositeEncoder(nn.Module):
    __metaclass__ = ABCMeta
    def __init__(self, args, encoder_1, embeddings_1, encoder_2, embeddings_2):
        super().__init__()
        self.args = args
        args.embedding_size = embeddings_1.shape[1]
        self.encoder_1 = encoder_1(args, embeddings_1)

        args.embedding_size = embeddings_2.shape[1]
        self.encoder_2 = encoder_2(args, embeddings_2)

        self.fc = []

        lin = nn.Linear(self.args.fully_connected_layer, self.args.hidden_dim[0])
        self.add_module('layer_' + str(args.num_layers) + '_fc_'+str(args.hidden_dim[0]), lin)
        self.fc.append(lin)
        for i in range(1, len(args.hidden_dim)):
            lin = nn.Linear(self.args.hidden_dim[i-1], self.args.hidden_dim[i])
            self.add_module('layer_' + str(args.num_layers+i) + '_fc_'+str(args.hidden_dim[i]), lin)
            self.fc.append(lin)

        self.dropout = nn.Dropout(args.dropout)
        self.hidden = nn.Linear(self.args.hidden_dim[-1], self.args.num_class)

    def forward(self, x_indx, mask=None):
        x_indx_1 = x_indx[0]
        x_indx_2 = x_indx[1]

        out1 = self.encoder_1(x_indx_1, fc=False)
        out2 = self.encoder_2(x_indx_2, fc=False)

        hidden = out1 + out2

        for i in range(len(self.fc)):
            hidden = self.fc[i](hidden)
            
        hidden = self.dropout(hidden)
        logit = self.hidden(hidden)

        return hidden, logit
