import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np

class OrnateReplicaModel(nn.Module):

    def __init__(self, num_retype=15, activation="elu", final_activation="sigmoid"):
        self.num_retype = num_retype
        self.activation = activation
        self.final_activation = final_activation
        self.CONV1 = 20
        self.CONV2 = 30
        self.CONV3 = 20
        self.NB_TYPE = 167

    def conv1(self, stride=1):
        return nn.Conv3d(self.num_retype, self.CONV1, kernel_size=3, padding=0,
                stride=stride, bias=False)

    def conv2(self, stride=1):
        return nn.Conv3d(self.CONV1, self.CONV2, kernel_size=4, padding=0,
                stride=stride, bias=False)

    def conv3(self,stride=1):
        return nn.Conv3d(self.CONV2, self.CONV3, kernel_size=4, padding=0,
                stride=stride, bias=False)

    def forward(self, features, score):

        prev_layer = torch.reshape(features, (-1, 167, 24, 24, 24))
        #TODO: Figure out retyper variable, temp soln make matrix 167 x 15
        retyper = np.zeros((167, 15))
        prev_layer = prev_layer.premute(0, 2, 3, 4, 1)
        
