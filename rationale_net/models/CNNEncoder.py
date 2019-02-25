import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
from rationale_net.models import AbstractEncoder

@RegisterModel('cnn')
def CNNEncoder(AbstractEncoder):
    def __init__(self, args, embeddings):
        super().__init__(args)
        
        if self.args.use_embedding_fc:
            self.embedding_fc = nn.Linear(hidden_dim, hidden_dim)
            self.embedding_bn = nn.BatchNorm1d(hidden_dim)
        
        
        self.cnn = cnn.CNN(args, max_pool_over_time = not args.use_as_tagger)
        self.lin = nn.Linear(len(self.args.filters)*self.args.filter_num[-1], self.args.hidden_dim[0])

    def forward(self, x, mask=None):
        x = super().forward(x)

        if mask is not None:        
            x = x * mask.unsqueeze(-1)
            
        if self.args.use_embedding_fc:
            x = F.relu( self.embedding_fc(x))
            x = self.dropout(x)
        
         
        x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
        hidden = self.cnn(x)
        hidden = hidden.view(hidden.size(0), -1)

        hidden = self.lin(hidden)

        return self.output(hidden) 
