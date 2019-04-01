import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from rationale_net.models.factory import RegisterModel, get_model
import copy

@RegisterModel('composite')
class CompositeEncoder(nn.Module):
    def __init__(self, args, char_embeddings, word_embeddings):
        super().__init__()
        
        assert len(args.encoders) == len(args.input_types)
        self.args = args
        
        self.encoders = []
        for i in range(len(args.encoders)):
            args_copy = copy.deepcopy(args)
            args_copy.embedding_size = char_embeddings.shape[1] if args.input_types[i] == 'char' else word_embeddings.shape[1]
            args_copy.model_form = args.encoders[i]
            args_copy.representation_type = args.input_types[i] 
            encoder = get_model(args_copy, char_embeddings, word_embeddings)
            self.add_module('encoder_' + str(i) + '_' + args.encoders[i], encoder)
            self.encoders.append(encoder)


        self.fc = []
        lin = nn.Linear(self.args.fully_connected_layer*len(args.encoders), self.args.hidden_dim[0])
        self.add_module('layer_' + str(args.num_layers) + '_fc_'+str(args.hidden_dim[0]), lin)
        self.fc.append(lin)

        for i in range(1, len(args.hidden_dim)):
            lin = nn.Linear(self.args.hidden_dim[i-1], self.args.hidden_dim[i])
            self.add_module('layer_' + str(args.num_layers+i) + '_fc_'+str(args.hidden_dim[i]), lin)
            self.fc.append(lin)

        self.dropout = nn.Dropout(args.dropout)
        self.hidden = nn.Linear(self.args.hidden_dim[-1], self.args.num_class)

    def forward(self, x_indx_word, x_indx_char, mask=None):
        hidden = torch.cat([self.encoders[i](x_char=x_indx_char, x_word=None, mask=mask, fc=False) if self.args.input_types[i] == 'char' \
            else self.encoders[i](x_char=None, x_word=x_indx_word, mask=mask, fc=False) for i in range(len(self.encoders))], dim=1)

        for i in range(len(self.fc)):
            hidden = self.fc[i](hidden)
            
        hidden = self.dropout(hidden)
        logit = self.hidden(hidden)

        return logit, hidden
