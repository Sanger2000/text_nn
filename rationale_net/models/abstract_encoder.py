from abc import ABCMeta, abstractmethod, abstractproperty
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

class AbstractEncoder(nn.Module):
    __metaclass__ = ABCMeta
    def __init__(self, args, fully_connected_shape, embeddings=None):
        super().__init__()
        self.args = args

        if self.args.pretrained_embedding and embeddings is not None:
            vocab_size, hidden_dim = embeddings.shape
            self.embedding_dim = hidden_dim
            self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)
            self.embedding_layer.weight.data = torch.from_numpy(embeddings)
            self.embedding_layer.weight.requires_grad = False
            if self.args.train_embedding:
                self.embedding_layer.weight.requires_grad = True
        else:
            vocab_size = self.args.vocab_size
            hidden_dim = self.args.embedding_size
            self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)

        if self.args.use_embedding_fc:
            self.embedding_fc = nn.Linear(hidden_dim, hidden_dim)
            self.embedding_bn = nn.BatchNorm1d(hidden_dim)

        self.lin = nn.Linear(fully_connected_shape, self.args.fully_connected_layer)
        self.fc = []
        if len(args.hidden_dim) != 0:
            lin = nn.Linear(self.args.fully_connected_layer, self.args.hidden_dim[0])
            self.add_module('layer_' + str(args.num_layers) + '_fc_'+str(args.hidden_dim[0]), lin)
            self.fc.append(lin)

        for i in range(1, len(args.hidden_dim)):
            lin = nn.Linear(self.args.hidden_dim[i-1], self.args.hidden_dim[i])
            self.add_module('layer_' + str(args.num_layers+i) + '_fc_'+str(args.hidden_dim[i]), lin)
            self.fc.append(lin)

        self.dropout = nn.Dropout(args.dropout)
        self.hidden = nn.Linear(self.args.hidden_dim[-1], self.args.num_class)

    def forward(self, x_indx):
        x = self.embedding_layer(x_indx.squeeze(1))
        if self.args.cuda:
            x = x.cuda()

        if self.args.use_embedding_fc:
            x = F.relu(self.embedding_fc(x))
            x = self.dropout(x)

        return x

    def output(self, x, fc=True):
        x = x.view(x.size(0), -1)
        hidden = self.lin(x)
        if not fc:
            return hidden

        for i in range(len(self.fc)):
            hidden = self.fc[i](hidden)
        hidden = self.dropout(hidden)
        logit = self.hidden(hidden)

        return logit, hidden
