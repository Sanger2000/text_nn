import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from rationale_net.models.factory import RegisterModel
from rationale_net.models.abstract_encoder import AbstractEncoder
from rationale_net.models.transformer_modules import PositionalEncoder, EncoderLayer, get_clones

class Concatenation(AbstractEncoder):
    def __init__(self, args, embeddings_1, embeddings_2):
        self.num_classes = self.args.num_classes
    
        
