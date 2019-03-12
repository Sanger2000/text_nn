import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from rationale_net.models.fully_connected import FullyConnected
from rationale_net.models.factory import RegisterModel
from rationale_net.models.cnn_encoder import CNNEncoder
import pdb

@RegisterModel('composite')
class CompositeEncoder(nn.Module):
    def __init__(self, args, word_embeddings, char_embeddings):
        super().__init__()
        self.args = args
        args.embedding_size = char_embeddings.shape[1]
        self.char_encoder = CNNEncoder(args, char_embeddings)

        args.embedding_size = word_embeddings.shape[1]
        self.word_encoder = CNNEncoder(args, word_embeddings)
       
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
        x_indx_char = x_indx[0]
        x_indx_word = x_indx[1]

        out1 = self.char_encoder(x_indx_char, fc=False)
        out2 = self.word_encoder(x_indx_word, fc=False)
        
        hidden = out1 + out2

        for i in range(len(self.fc)):
            hidden = self.fc[i](hidden)
        hidden = self.dropout(hidden)
        logit = self.hidden(hidden)
        
        return hidden, logit

	
