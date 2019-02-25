from abc import ABCMeta abstractmethod, abstractproperty
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

class AbstractEncoder(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, args):
        super().__init__()
        self.args = args

        if self.args.pretrained_embedding:
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
        
        for i in range(len(args.hidden_dim)):
            lin = nn.Linear(self.args.hidden_dim[i-1], self.args.hidden_dim[i])
            self.add_module('layer_' + str(args.num_layers+i) + '_fc_'+str(args.hidden_dim[i]), lin)
            self.fc.append(lin)

        self.dropout = nn.Dropout(args.dropout)
        self.hidden = nn.Linear(self.args.hidden_dim[-1], self.args.num_class)
    
    def forward(self, x):
        x = self.embedding_layer(x_indx.squeeze(1))
        if self.args.cuda:
                x = x.cuda()
        return x
    
    def output(self, x):
        hidden = x
        if len(self.fc) != 0:
            for i in range(len(self.fc)):
                hidden = self.fc[i](hidden)

        hidden = self.dropout(hidden)
        logit = self.hidden(hidden)
        return logit, hidden
