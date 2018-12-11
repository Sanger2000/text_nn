import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

class CNN(nn.Module):

    def __init__(self, args, max_pool_over_time=False):
        super(CNN, self).__init__()
        '''
        self.args = args
        self.layers = []
        for layer in range(self.args.num_layers):
            if self.args.kernel_sizes[layer] != None:
                self.add_char_conv(layer)
            if self.args.pool_sizes[layer] != None:
                self.add_max_pooling(layer)
            if self.args.filter_num != None:
                self.add_word_conv(layer)

        self.max_pool = max_pool_over_time
        '''
        self.args = args
        #model after the word level CNN
        #make it like 3, 5, 7 kernel sizes with padding that outputs 256 output dims
        #make another 3, 5, 7 kernel sizes with padding that ouputs 1024 output dims
        #concatenate, then run Global Max Pooling
        #then run fully connected layer.
        self.conv1 = nn.Conv1d(in_channels=self.args.embedding_dim, 
        self.conv2 = nn.Conv1d(in_channels=self.args.filter_sizes[0], out_channels=self.args.filter_sizes[1], kernel_size=7)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=self.args.filter_sizes[1], out_channels=self.args.filter_sizes[2], kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=self.args.filter_sizes[2], out_channels=self.args.filter_sizes[3], kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=self.args.filter_sizes[3], out_channels=self.args.filter_sizes[4], kernel_size=3)
        self.conv6 = nn.Conv1d(in_channels=self.args.filter_sizes[4], out_channels=self.args.filter_sizes[5], kernel_size=3)
        self.pool6 = nn.MaxPool1d(kernel_size=3)
    def add_word_conv(self, layer):
        convs = []
        for filt in self.args.filters:
            in_channels =  self.args.embedding_dim if layer == 0 else self.args.filter_num * len( self.args.filters)
            kernel_size = filt
            new_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.args.filter_num, kernel_size=kernel_size)
            self.add_module( 'layer_'+str(layer)+'_conv_'+str(filt), new_conv)
            convs.append(new_conv)

        self.layers.append(conv)

    def add_char_conv(self, layer, activ=nn.ReLU()):
        in_channels = self.args.embedding_dim if layer == 0 else self.args.filter_sizes[layer-1]
        kernel_size = self.args.kernel_sizes[layer]
        conv = nn.Conv1d(in_channels=in_channels, out_channels = self.args.filter_sizes[layer], kernel_size=kernel_size)

        self.add_module('layer_' + str(layer) + '_conv_' + str(kernel_size), conv)
        self.layers.append(conv)

        if activ != None:
            self.add_module('layer_' + str(layer) + "_activ_" + str(activ), activ)
            self.layers.append(activ)

    def add_max_pooling(self, layer):
        kernel_size = self.args.pool_sizes[layer]
        pool_layer = nn.MaxPool1d(kernel_size=kernel_size)

        self.add_module('layer_' + str(layer) + "_pool_" + str(kernel_size), pool_layer)
        self.layers.append(pool_layer)

    def _conv(self, x):
        layer_activ = x
        for layer in self.layers:
            next_activ = []
            for conv in layer:
                left_pad = conv.kernel_size[0] - 1
                pad_tensor_size = [d for d in layer_activ.size()]
                pad_tensor_size[2] = left_pad
                left_pad_tensor =autograd.Variable( torch.zeros( pad_tensor_size ) )
                if self.args.cuda:
                    left_pad_tensor = left_pad_tensor.cuda()
                padded_activ = torch.cat( (left_pad_tensor, layer_activ), dim=2)
                next_activ.append( conv(padded_activ) )

            # concat across channels
            layer_activ = F.relu( torch.cat(next_activ, 1) )

        return layer_activ


    def _pool(self, relu):
        pool = F.max_pool1d(relu, relu.size(2)).squeeze(-1)
        return pool


    def forward(self, x):
        ''' 
        if self.args.filters != None:
            activ = self._conv(x)
            if self.max_pool:
                activ = self.pool(next_activ)
            return next_activ

        for layer in self.layers:
            x = layer(x)
        return x
        '''
        activ = self.conv1(x)
        activ = F.relu(activ)
        activ = self.pool1(activ)
        activ = self.conv2(activ)
        activ = F.relu(activ)
        activ = self.pool2(activ)
        activ = self.conv3(activ)
        activ = F.relu(activ)
        activ = self.conv4(activ)
        activ = F.relu(activ)
        activ = self.conv5(activ)
        activ = F.relu(activ)
        activ = self.conv6(activ)
        activ = F.relu(activ)
        activ = self.pool6(activ)
        return activ
