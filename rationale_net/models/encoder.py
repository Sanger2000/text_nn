import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
import pdb

class Encoder(nn.Module):

    def __init__(self, embeddings, args):
        super(Encoder, self).__init__()
        ### Encoder
        self.args = args
        vocab_size, hidden_dim = embeddings.shape
        self.embedding_dim = hidden_dim
        self.embedding_layer = nn.Embedding( vocab_size, hidden_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = True
        self.embedding_fc = nn.Linear( hidden_dim, hidden_dim )
        self.embedding_bn = nn.BatchNorm1d( hidden_dim)

        if args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time=(not args.use_as_tagger))

        else:
            raise NotImplementedError("Model form {} not yet supported for encoder!".format(args.model_form))

        self.dropout = nn.Dropout(args.dropout)
        self.hidden = nn.Linear(args.hidden_dim, args.num_class)

    def forward(self, x_indx, mask=None):
        '''
            x_indx:  batch of word indices
            mask: Mask to apply over embeddings for tao rationales
        '''
        x = self.embedding_layer(x_indx.squeeze(1))
        if self.args.cuda:
                x = x.cuda()
        if not mask is None:
            print(x.shape)
            x = x * mask.unsqueeze(-1)
            print(x.shape)
        if self.args.use_embedding_fc:
            x = F.relu( self.embedding_fc(x))
            x = self.dropout(x)

        if self.args.model_form == 'cnn':
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            hidden = self.cnn(x)
            hidden = hidden.view(hidden.size(0), -1)
            self.fc = nn.Linear(hidden.size(1), self.args.hidden_dim)

            hidden = F.relu( self.fc(hidden) )
        else:
            raise Exception("Model form {} not yet supported for encoder!".format(args.model_form))

        hidden = self.dropout(hidden)
        logit = self.hidden(hidden)
        return logit, hidden
