import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from rationale_net.models.factory import RegisterModel
from rationale_net.models.abstract_encoder import AbstractEncoder
import pdb

@RegisterModel('rnn')
class BidirectionalLSTMEncoder(AbstractEncoder):

    def __init__(self, args, embeddings):
        super().__init__(args, args.d_ff*2, embeddings)
        
        self.lstm = nn.LSTM(args.embedding_size, args.d_ff, args.num_layers, \
                                batch_first=True, bidirectional=True)

    def forward(self, x, mask=None):
        x = super().forward(x)

        h0 = torch.zeros(self.args.num_layers*2, x.size(0), self.args.d_ff)
        c0 = torch.zeros(self.args.num_layers*2, x.size(0), self.args.d_ff)

        if self.args.cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        out, _ = self.lstm(x, (h0, c0))
        return super().output(out[:, -1, :])

