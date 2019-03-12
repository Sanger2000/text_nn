import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from rationale_net.models.fully_connected import FullyConnected
from rationale_net.models.factory import RegisterModel
from rationale_net.models.cnn_encoder import CNNEncoder
from rationale_net.models.abstract_composite_encoder import abstract_composite_encoder
import pdb

@RegisterModel('composite-char-word')
class CompositeEncoder(AbstractCompositeEncoder):
    def __init__(self, args, word_embeddings, char_embeddings):
        super().__init__(args, CNNEncoder, char_embeddings, CNNEncoder, word_embeddings)

    def forward(self, x_indx, mask=None):
        return super().forward(x_indx, mask)
