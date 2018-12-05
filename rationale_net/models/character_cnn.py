import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pdb

NUM_CONV_LAYERS = 6
NUM_FULLY_CONNECTED_LAYERS = 3
LARGE_FILTER_SIZE = 1024
SMALL_FILTER_SIZE = 256
LARGE_FULLY_CONNECTED_SIZE = 2048
SMALL_FULLY_CONNECTED_SIZE = 1024
KERNEL_SIZES = [7, 7, 3, 3, 3, 3]
MAX_POOL_SIZES = [3, 3, None, None, None, 3]
MAX_SEQUENCE_LENGTH=1014

class CharacterCNN(nn.Module):

    def __init__(self, args):
        super(CharacterCNN, self).__init__()

        self.args = args

        self.filter_size = SMALL_FILTER_SIZE
        self.fully_connected_size = SMALL_FULLY_CONNECTED_SIZE

        if self.args.use_large:
            self.filter_size = LARGE_FILTER_SIZE
            self.fully_connected_size = LARGE_FULLY_CONNECTED_SIZE

        self.total_size = args.max_sequence_length

        self.conv_layers = []
        self.pool_layers = []
        self.connected_layers = []
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        for layer in range(NUM_CONV_LAYERS):
            in_channel = args.embedding_dim if layer==0 else self.filter_size:
            kernel_size = KERNEL_SIZES[layer]

            new_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.filter_size, kernel_size=kernel_size)
            self.add_module( 'layer_'+str(layer), new_conv)
            self.layers.append(new_conv)

            self.pool_layers.append(None)
            if MAX_POOL_SIZES[layer] != None:
                self.pool_layers[-1] = nn.MaxPool1d(kernel_size=MAX_POOL_SIZES[layer])



    def forward(self, x):
        #layers 1-6
        for layer in range(len(self.conv_layers)):
            x = self.conv_layers[layer](x)
            x = self.relu(x)
            if self.pool_layers[layer] != None:
                x = self.pool_layers[layers](x)

        #flatten before fully connected layer
        x = x.view(x.size(0), -1)

        #layer 7:
        x = nn.Linear(x.size(1), self.fully_connected_size)(x)
        x = self.relu(x)
        x = self.dropout(x)

        #layer 8:
        x = nn.Linear(self.fully_connected_size, self.fully_connected_size)(x)
        x = self.relu(x)
        x = self.dropout()

        #layer 9:
        x = nn.Linear(self.fully_connected_size, args.output_layer)()
        x = self.relu(x)

        return x
