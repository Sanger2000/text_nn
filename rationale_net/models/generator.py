import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
from rationale_net.models.transformer_modules import PositionalEncoder, EncoderLayer, Norm, get_clones
import rationale_net.utils.learn as learn
import pdb

'''
    The generator selects a rationale z from a document x that should be sufficient
    for the encoder to make it's prediction.

    Several froms of Generator are supported. Namely CNN with arbitary number of layers, and @taolei's FastKNN
'''
class Generator(nn.Module):

    def __init__(self, embeddings, args):
        super(Generator, self).__init__()

        self.args = args
        if self.args.pretrained_embedding:
            vocab_size, hidden_dim = embeddings.shape
            self.embedding_dim = hidden_dim
            self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)
            self.embedding_layer.weight.data = torch.from_numpy(embeddings)
            self.embedding_layer.weight.requires_grad = False
            if self.args.train_embedding:   
                self.embedding_layer.weight.requires_grad = True    
        else:
            vocab_size = self.args.max_word_length	
            hidden_dim = self.args.d_model
            self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)

        if self.args.use_embedding_fc:
            self.embedding_fc = nn.Linear(hidden_dim, hidden_dim)
            self.embedding_bn = nn.BatchNorm1d(hidden_dim)

        if args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time=True)
            lin = nn.Linear(len(self.args.filters)*self.args.filter_num[-1], self.args.hidden_dim[0])

        elif args.model_form == 'transformer':
            
            self.N = self.args.N
            d_model = self.args.d_model
                
            self.pe = PositionalEncoder(self.args.cuda, self.args.max_word_length, d_model)
            self.layers = get_clones(EncoderLayer(args), self.N)
            self.norm = Norm(d_model, self.args.eps)
            

            lin = nn.Linear(self.args.d_model*vocab_size, self.args.hidden_dim[0])

        else:
            raise NotImplementedError("Model form {} not yet supported for encoder!".format(args.model_form))
                

        self.z_dim = 2


        self.dropout = nn.Dropout(args.dropout)


    def  __z_forward(self, activ):
        '''
            Returns prob of each token being selected
        '''
        activ = activ.transpose(1, 2)
        logits = self.hidden(activ)
        probs = learn.gumbel_softmax(logits, self.args.gumbel_temprature, self.args.cuda)
        z = probs[:,:,1]
        return z


    def forward(self, x_indx):
        '''
            Given input x_indx of dim (batch, length), return z (batch, length) such that z
            can act as element-wise mask on x
        '''
        x = self.embedding_layer(x_indx.squeeze(1))
        if self.args.cuda():
            x = x.cuda()
        
        if self.args.model_form == 'cnn':
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            activ = self.cnn(x)

        elif self.args.model_form == 'transformer':
            x = self.pe(x)
            for i in range(N):
                x = self.layers[i](x, mask)
            activ = self.norm(x)
        else:
            raise NotImplementedError("Model form {} not yet supported for generator!".format(args.model_form))


        z = self.__z_forward(F.relu(activ))
        mask = self.sample(z)
        return mask, z


    def sample(self, z):
        '''
            Get mask from probablites at each token. Use gumbel
            softmax at train time, hard mask at test time
        '''
        mask = z
        if self.training:
            mask = z
        else:
            ## pointwise set <.5 to 0 >=.5 to 1
            mask = learn.get_hard_mask(z)
        return mask


    def loss(self, mask, x_indx):
        '''
            Compute the generator specific costs, i.e selection cost, continuity cost, and global vocab cost
        '''
        selection_cost = torch.mean( torch.sum(mask, dim=1) )
        l_padded_mask =  torch.cat( [mask[:,0].unsqueeze(1), mask] , dim=1)
        r_padded_mask =  torch.cat( [mask, mask[:,-1].unsqueeze(1)] , dim=1)
        continuity_cost = torch.mean( torch.sum( torch.abs( l_padded_mask - r_padded_mask ) , dim=1) )
        return selection_cost, continuity_cost

class CharacterGenerator(nn.Module):
    def __init__(self, embeddings, args):
        super(Generator, self).__init__()
        vocab_size, hidden_dim = embeddings.shape
        self.embedding_layer = nn.Embedding( vocab_size, hidden_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.weight.requires_grad = False
        self.args = args

        if args.model_form == 'char_cnn':
            self.cnn = char_cnn.CharacterCNN(args)

    def  __z_forward(self, activ):
        '''
            Returns prob of each token being selected
        '''
        probs = learn.gumbel_softmax(activ, self.args.gumbel_temprature, self.args.cuda)
        z = probs[:,:, 1]
        return z


    def forward(self, x_indx):
        '''
            Given input x_indx of dim (batch, length), return z (batch, length) such that z
            can act as element-wise mask on x
        '''
        if self.args.model_form == 'cnn':
            x = self.embedding_layer(x_indx.squeeze(1))
            if self.args.cuda:
                x = x.cuda()
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            activ = self.cnn(x)
        else:
            raise NotImplementedError("Model form {} not yet supported for generator!".format(args.model_form))

        z = self.__z_forward(F.relu(activ))
        mask = self.sample(z)
        return mask, z


    def sample(self, z):
        '''
            Get mask from probablites at each token. Use gumbel
            softmax at train time, hard mask at test time
        '''
        mask = z
        if self.training:
            mask = z
        else:
            ## pointwise set <.5 to 0 >=.5 to 1
            mask = learn.get_hard_mask(z)
        return mask


    def loss(self, mask, x_indx):
        '''
            Compute the generator specific costs, i.e selection cost, continuity cost, and global vocab cost
        '''
        selection_cost = torch.mean( torch.sum(mask, dim=1) )
        l_padded_mask =  torch.cat( [mask[:,0].unsqueeze(1), mask] , dim=1)
        r_padded_mask =  torch.cat( [mask, mask[:,-1].unsqueeze(1)] , dim=1)
        continuity_cost = torch.mean( torch.sum( torch.abs( l_padded_mask - r_padded_mask ) , dim=1) )
        return selection_cost, continuity_cost
