import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from rationale_net.models.factory import RegisterModel
from rationale_net.models.abstract_encoder import AbstractEncoder
from rationale_net.models import cnn

@RegisterModel('cnn')
class CNNEncoder(AbstractEncoder):
    def __init__(self, args, embeddings):
        super().__init__(args, len(args.filters)*args.filter_num[-1], embeddings)
        
        self.cnn = cnn.CNN(args, max_pool_over_time = not args.use_as_tagger)

    def forward(self, x_char=None, x_word=None, mask=None, fc=True):
        x = super().forward(x_char, x_word)

        if mask is not None:        
            x = x * mask.unsqueeze(-1)
            
        x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
        hidden = self.cnn(x)

        return super().output(hidden, fc)
