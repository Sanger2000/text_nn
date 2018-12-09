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
        if self.args.use_embedding_fc:
            self.embedding_layer.weight.requires_grad = True
            self.embedding_fc = nn.Linear(hidden_dim, hidden_dim)
            self.embedding_bn = nn.BatchNorm1d(hidden_dim)
        if args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time=False)
            self.flat_dim =  len(args.filters)*args.filter_num
            if isChar:
                self.flat_dim = int((self.max_sequence_length-96)/27)
        else:
            raise NotImplementedError("Model form {} not yet supported for encoder!".format(args.model_form))


        self.fully_connected = []
        for layer in range(len(self.args.hidden_dim)+1):
            if layer == 0:
                self.add_linear_layer(self.flat_dim, self.args.hidden_dim[layer])

            elif layer == len(self.args.hidden_dim):
                self.add_linear_layer(args.hidden_dim[layer-1], args.num_class)

            else:
                self.add_linear_layer(args.hidden_dim[layer-1], args.hidden_dim[layer])



    def add_linear_layer(self, input_size, output_size):
        self.fully_connected.append(nn.Linear(input_size, output_size))

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

        if self.args.use_embedding_fc:
            x = F.relu( self.embedding_fc(x))
            x = self.dropout(x)


        if self.args.model_form == 'cnn':
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            hidden = self.cnn(x)
            hidden = hidden.view(hidden.size(0), -1)

            for layer in range(len(self.fully_connected)-1):
                hidden  = self.dropout(F.relu(layer(hidden))

        else:
            raise Exception("Model form {} not yet supported for encoder!".format(args.model_form))

        logit = self.fully_connected[-1](hidden)
        return logit, hidden
