import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from rationale_net.models.transfomer_modules import Embedder, PositionalEncoder, EncoderLayer, DecoderLayer, get_clones
import pdb

class Encoder(nn.Module):
    def __init__(self, args, vocab_size, embeddings=None):
        super().__init__()
        self.args = args

        self.N = self.args.N
        d_model = self.args.d_model

        if embeddings != None:
            self.embed = nn.Embedding()
            self.embed.weight.data = torch.from_numpy(embeddings)
            self.embed.weight.requires_grad = False
        else:
            self.embed = Embedder(vocab_size, d_model)
            
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, self.args.heads), self.N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, args, vocab_size, embeddings=None):
        super().__init__()
        self.args = args

        self.N = self.args.N
        d_model = self.args.d_model 
        
        if embeddings != None:
            self.embed = nn.Embedding()
            self.embed.weight.data = torch.from_numpy(embeddings)
            self.embed.weight.requires_grad = False
        else:
            self.embed = Embedder(vocab_size, d_model)

        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, self.args.heads), self.N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        d_model = self.args.d_model

        src_vocab = self.args.src_vocab
        trg_vocab = self.args.trg_vocab
        self.encoder = Encoder(args, src_vocab)
        self.decoder = Decoder(args, trg_vocab)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
