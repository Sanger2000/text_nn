

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
import pdb

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):

    def __init__(self, args, vocab_size, d_model):
        super(PositionalEncoder, self).__init__()
        self.args = args
        self.d_model = d_model
        self.pe = numpy.zeroes(vocab_size, self.d_model)
        for pos in range(vocab_size):
            for i in range(0, self.d_model, 2):
                self.pe[pos, i] = math.sin(pos / (10000 ** (2*i/self.d_model)))
                self.pe[pos, i+1] = math.cos(pos / (10000 ** (2*i/self.d_model)))

        pe.unsqueeze(0)


    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        
        var = Variable(self.pe[:, :seq_len], requires_grad=False)
        if self.args.cuda:
            var.cuda()
        x = x + var
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, args, embeddings):
        super(MultiHeadAttention, self).__init__()
        
        self.args = args
        self.heads = self.args.heads
        self.vocab_size, self.hidden_dim = embeddings.shape
        self.d_k = self.hidden_dim // self.heads

        self.q_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.dropout = nn.Dropout(self.args.dropout)
        self.out = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, q, v, k, mask=None):
        batch_size = q.size(0)

        k = self.k_linear(k).view(batch_size, -1, self.heads, self.d_k)
        q = self.k_linear(q).view(batch_size, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.heads, seld.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        concat = scores.transpose(1,2).contiguous.view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
    def attention(self, q, k, v, d_k, mask, droout):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
    
        if dropout is not None:
            scores = dropout(scores)
            
        output = torch.matmul(scores, v)
        return output

class FeedForward(nn.Module):
    def __init__(self, args, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.args = args

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(self.args.dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, args, d_model):
        super(Norm, self).__init__()

        self.size = d_model
        
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, args): 
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttentioni(self.args.heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(self.args.dropout)
        self.dropout_2 = nn.Dropout(self.args.dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, args,  d_model):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(self.args.dropout)
        self.dropout_2 = nn.Dropout(self.args.dropout)
        self.dropout_3 = nn.Dropout(self.args.dropout)

        self.attn_1 = MultiHeadAttention(self.args.heads, d_model)
        self.attn_2 = MultiHeadAttention(self.args.heads, d_model)
        self.ff = FeedForward(d_model)
        if self.args.cuda:
            self.ff.cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
