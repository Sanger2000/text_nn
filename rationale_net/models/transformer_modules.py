import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
import pdb
import copy


class PositionalEncoder(nn.Module):

    def __init__(self, cuda, max_seq_len, embedding_size):
        super(PositionalEncoder, self).__init__()
        self.cuda = cuda
        self.embedding_size = embedding_size
        self.pe = torch.zeros(max_seq_len, self.embedding_size)
        for pos in range(max_seq_len):
            for i in range(0, self.embedding_size, 2):
                self.pe[pos, i] = math.sin(pos / (10000 ** (2*i/self.embedding_size)))
                self.pe[pos, i+1] = math.cos(pos / (10000 ** (2*i/self.embedding_size)))

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)

        pe = autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        if self.cuda:
            pe = pe.cuda()
        
        output = torch.cat((x, pe.expand(x.size(0), pe.size(1), pe.size(2))), dim=-1) 
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout):
        super(MultiHeadAttention, self).__init__()
        
        self.d_split = d_model // heads
        self.heads = heads
        self.d_model = d_model
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
            
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        k = self.k_linear(x).view(batch_size, -1, self.heads, self.d_split)
        q = self.k_linear(x).view(batch_size, -1, self.heads, self.d_split)
        v = self.v_linear(x).view(batch_size, -1, self.heads, self.d_split)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_split, mask, self.dropout)
        
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out(concat)
	
        return output
    
    def attention(self, q, k, v, d_split, mask, dropout):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_split)

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
        hidden = self.dropout(F.relu(self.linear_1(x)))
        output = self.linear_2(hidden)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, args): 
        super(EncoderLayer, self).__init__()
        self.args = args

        d_model = self.args.d_model
        self.norm_1 = nn.LayerNorm((self.args.max_word_length, d_model)
        self.norm_2 = nn.LayerNorm((self.args.max_word_length, d_model)

        self.attn = MultiHeadAttention(self.args.heads, d_model, self.args.dropout)
        self.ff = FeedForward(d_model, self.args.d_ff, self.args.dropout)
        self.dropout_1 = nn.Dropout(self.args.dropout)
        self.dropout_2 = nn.Dropout(self.args.dropout)
        
    def forward(self, x, mask):
        normalized_inp = self.norm_1(x)
        next_layer = x + self.dropout_1(self.attn(normalized_inp, mask))
        normalized_next_layer = self.norm_2(next_layer)
        output = next_layer + self.dropout_2(self.ff(normalized_next_layer))

        return output

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
