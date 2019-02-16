import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
import pdb
import copy


class PositionalEncoder(nn.Module):

    def __init__(self, cuda, max_seq_len, d_model):
        super(PositionalEncoder, self).__init__()
        self.cuda = cuda
        self.d_model = d_model
        self.pe = torch.zeros(max_seq_len, self.d_model)
        for pos in range(max_seq_len):
            for i in range(0, self.d_model, 2):
                self.pe[pos, i] = math.sin(pos / (10000 ** (2*i/self.d_model)))
                self.pe[pos, i+1] = math.cos(pos / (10000 ** (2*i/self.d_model)))

        self.pe = self.pe.unsqueeze(0)


    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        
        seq_len = x.size(1)


        pe = autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        if self.cuda:
            pe = pe.cuda()
            #concatenate x and pe

        x = x + pe
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout):
        super(MultiHeadAttention, self).__init__()
        
        self.d_k = d_model // heads
        self.heads = heads
        self.d_model = d_model
        
        #make instead head different matrices
        #split them up
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, v, k, mask=None):
        batch_size = q.size(0)

        k = self.k_linear(k).view(batch_size, -1, self.heads, self.d_k)
        q = self.k_linear(q).view(batch_size, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.heads, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out(concat)
	
        return output
    
    def attention(self, q, k, v, d_k, mask, dropout):
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
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
#use pytorch layer Norm instead
class Norm(nn.Module):
    def __init__(self, d_model, eps):
        super(Norm, self).__init__()

        self.size = d_model
        
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, args): 
        super().__init__()
        self.args = args

        d_model = self.args.d_model
        self.norm_1 = Norm(d_model, self.args.eps)
        self.norm_2 = Norm(d_model, self.args.eps)

        self.attn = MultiHeadAttention(self.args.heads, d_model, self.args.dropout)
        self.ff = FeedForward(d_model, self.args.d_ff, self.args.dropout)
        self.dropout_1 = nn.Dropout(self.args.dropout)
        self.dropout_2 = nn.Dropout(self.args.dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        #make variable names more descriptive
        #use one input into multihead attention
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        d_model = self.args.d_model

        self.norm_1 = Norm(d_model, self.args.eps)
        self.norm_2 = Norm(d_model, self.args.eps)
        self.norm_3 = Norm(d_model, self.args.eps)

        self.dropout_1 = nn.Dropout(self.args.dropout)
        self.dropout_2 = nn.Dropout(self.args.dropout)
        self.dropout_3 = nn.Dropout(self.args.dropout)

        self.attn_1 = MultiHeadAttention(self.args.heads, d_model, self.args.dropout)
        self.attn_2 = MultiHeadAttention(self.args.heads, vocab_size, d_model, self.args.dropout)
        self.ff = FeedForward(d_model, self.args.d_ff, self.args.dropout)

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
