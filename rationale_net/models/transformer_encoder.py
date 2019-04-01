import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from rationale_net.models.factory import RegisterModel
from rationale_net.models.abstract_encoder import AbstractEncoder
from rationale_net.models.transformer_modules import PositionalEncoder, EncoderLayer, get_clones

@RegisterModel('transformer')
class TransformerEncoder(AbstractEncoder):

    def __init__(self, args, embeddings):
        super().__init__(args, args.d_model*args.max_word_length, embeddings)
        
        self.N = self.args.N
        d_model = self.args.d_model
        embedding_size = self.args.embedding_size
            
        self.pe = PositionalEncoder(self.args.cuda, self.args.max_word_length, embedding_size)
        self.layers = get_clones(EncoderLayer(args), self.N)
        self.norm = nn.LayerNorm((self.args.max_word_length, d_model))


    def forward(self, x_char=None, x_word=None, mask=None):
        x = super().forward(x_char, x_word)
        
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, mask)
        hidden = self.norm(x)

        return self.output(hidden) 
