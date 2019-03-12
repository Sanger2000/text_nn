import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

class CNN(nn.Module):

    def __init__(self, args, max_pool_over_time=False, intermediary_dropout=False):
        super(CNN, self).__init__()
        print(args.filter_num)
        print(args.num_layers)
        self.args = args
        self.layers = []        
        for layer in range(self.args.num_layers):
            convs = []
            for filt in self.args.filters:
                in_channels = self.args.embedding_size if layer==0 else self.args.filter_num[layer-1] * len(self.args.filters)
                kernel_size = filt
                new_conv = nn.Conv1d(in_channels = in_channels, out_channels = args.filter_num[layer], kernel_size=kernel_size)
                self.add_module('layer_'+str(layer)+'_conv_'+str(filt), new_conv)
                convs.append(new_conv)
            self.layers.append(convs)
        self.max_pool = max_pool_over_time
        self.intermediary_dropout = intermediary_dropout
        self.dropout = nn.Dropout(self.args.dropout)

    def _conv(self, x):
        layer_activ = x
        for layer in self.layers:
            next_activ = []
            for conv in layer:
                left_pad = conv.kernel_size[0] - 1
                pad_tensor_size = [d for d in layer_activ.size()]
                pad_tensor_size[2] = left_pad
                left_pad_tensor = autograd.Variable( torch.zeros( pad_tensor_size ) )
                if self.args.cuda:
                    left_pad_tensor = left_pad_tensor.cuda()
                padded_activ = torch.cat( (left_pad_tensor, layer_activ), dim=2)
                next_activ.append( conv(padded_activ) )

            # concat across channels
            layer_activ = F.relu( torch.cat(next_activ, 1) )
            if self.intermediary_dropout:
                layer_activ = self.dropout(layer_activ)
        return layer_activ


    def _pool(self, relu):
        pool = F.max_pool1d(relu, relu.size(2)).squeeze(-1)
        return pool


    def forward(self, x):
        activ = self._conv(x)
        if self.max_pool:
            activ = self._pool(activ)
        return activ 
