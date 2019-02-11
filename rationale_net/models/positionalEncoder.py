
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
import pdb

class PositionalEncoder(nn.Module):

    def __init__(self, args, embeddings):
        super(PositionalEncoder, self).__init__()
        self.args = args
        self.vocab_size, self.hidden_dim = embeddings.shape
        self.pe = numpy.zeroes(embeddings.shape)
        for pos in range(self.vocab_size):
            for i in range(0, self.hidden_dim, 2):
                self.pe[pos, i] = math.sin(pos / (10000 ** (2*i/self.hidden_dim)))
                self.pe[pos, i+1] = math.cos(pos / (10000 ** (2*i/self.hidden_dim)))

        pe.unsqueeze(0)


    def forward(self, x):
        x = x * math.sqrt(self.hidden_dim)
        seq_len = x.size(1)
        
        var = Variable(self.pe[:, :seq_len], requires_grad=False)
        if self.args.cuda:
            var.cuda()
        x = x + var
        return x

