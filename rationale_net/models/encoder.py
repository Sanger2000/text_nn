import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
import pdb

class Encoder(nn.Module):

    def __init__(self, embeddings, args, isChar):
        super(Encoder, self).__init__()
        ### Encoder
        self.args = args
        vocab_size, hidden_dim = embeddings.shape
        self.embedding_dim = hidden_dim
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.weight.requires_grad = False
        '''
        if self.args.use_embedding_fc:
            self.embedding_layer.weight.requires_grad = True
            self.embedding_fc = nn.Linear(hidden_dim, hidden_dim)
            self.embedding_bn = nn.BatchNorm1d(hidden_dim)
        '''
        if args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time=False)
        else:
            raise NotImplementedError("Model form {} not yet supported for encoder!".format(args.model_form))
        '''
        self.fully_connected = []
        for layer in range(len(self.args.hidden_dim)+1):
            if layer == 0:
                self.fully_connected.append(self.add_linear_layer(self.flat_dim, self.args.hidden_dim[layer]).cuda())

            elif layer == len(self.args.hidden_dim):
                self.fully_connected.append(self.add_linear_layer(args.hidden_dim[layer-1], args.num_class).cuda())

            else:
                self.fully_connected.append(self.add_linear_layer(args.hidden_dim[layer-1], args.hidden_dim[layer]).cuda())
   
        self.dropout = nn.Dropout(self.args.dropout)
        self.fc = nn.Linear(self.flat_dim, self.args.hidden_dim[0])
        self.first_hidden = nn.Linear(self.args.hidden_dim[0], self.args.hidden_dim[1])
        self.second_hidden = nn.Linear(self.args.hidden_dim[1], self.args.num_class)
        self.dropout = nn.Dropout(self.args.dropout)
        '''
        self.fc = nn.Linear(50*256, self.args.hidden_dim[0])
        self.dropout = nn.Dropout(args.dropout) 
        self.first_hidden = nn.Linear(args.hidden_dim[0], args.hidden_dim[0])
        self.second_hidden = nn.Linear(args.hidden_dim[0], args.num_class)
    def add_linear_layer(self, input_size, output_size):
        return nn.Linear(input_size, output_size)

    def forward(self, x_indx, mask=None):
        '''
            x_indx:  batch of word indices
            mask: Mask to apply over embeddings for tao rationales
        '''
        x = self.embedding_layer(x_indx.squeeze(1))
        if self.args.cuda:
                x = x.cuda()
        if not mask is None:
            x = x * mask.unsqueeze(-1)
        '''
        if self.args.use_embedding_fc:
            x = F.relu( self.embedding_fc(x))
            x = self.dropout(x)
        '''

        if self.args.model_form == 'cnn':
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            hidden = self.cnn(x)
            hidden = hidden.view(hidden.size(0), -1)
            '''
            for layer in range(len(self.fully_connected)-1):
                hidden  = self.dropout(F.relu(self.fully_connected[layer](hidden)))
            '''
            hidden = F.relu(self.fc(hidden))
        else:
            raise Exception("Model form {} not yet supported for encoder!".format(args.model_form))
        hidden = self.dropout(hidden)
        hidden = self.first_hidden(hidden)
        hidden = self.dropout(hidden)
        logit = self.second_hidden(hidden)
        return logit, hidden
