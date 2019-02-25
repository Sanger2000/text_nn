import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from rationale_net.models.abstract_encoder import AbstractEncoder
from rationale_net.models.transformer_modules import PositionalEncoder, EncoderLayer, Norm, get_clones

@RegisterModel('transformer')
def TransformerEncoder(AbstractEncoder):
    def __init__(self, args, embeddings):
        super().__init__(args)
        
        self.cnn = cnn.CNN(args, max_pool_over_time = not args.use_as_tagger)
        self.lin = nn.Linear(len(self.args.filters)*self.args.filter_num[-1], self.args.hidden_dim[0])

        self.N = self.args.N
        d_model = self.args.d_model
        embedding_size = self.args.embedding_size
            
        self.pe = PositionalEncoder(self.args.cuda, self.args.max_word_length, embedding_size)
        self.layers = get_clones(EncoderLayer(args), self.N)
        self.norm = Norm(d_model, self.args.eps)
        

        self.lin = nn.Linear(self.args.d_model*self.args.max_word_length, self.args.hidden_dim[0])

    def forward(self, x, mask=None):
        x = super().forward(x)
        
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, mask)
        hidden = self.norm(x)
        hidden = self.lin(hidden)

        return self.output(hidden) 
