
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

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

