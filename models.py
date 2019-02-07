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

        prev_layer = torch.reshape(features, (-1, self.NB_TYPE, 24, 24, 24))

        # Retyper: a [high dimension * low dimension] tensor
        retyper_matrix = torch.randn(self.NB_TYPE, self.num_retype)

        # Rearrange dimensions so that the "167" (feature-vector for the point)
        # is in the innermost (rightmost) dimension
        prev_layer = prev_layer.permute(0, 2, 3, 4, 1)

        # Dimensions of the new tensor
        new_map_shape = prev_layer.shape
        new_map_shape[4] = self.num_retype

        # Reshape so that we have a two-dimensional tensor.
        # Each row will represent an (x,y,z) point, which has a 167-dimensional feature vector.
        prev_layer = torch.reshape(prev_layer, (-1, self.NB_TYPE))

        # Multiply each (x,y,z) point's feature vector by the retyper matrix,
        # to reduce the dimensionality of the feature vectors
        prev_layer = torch.matmul(prev_layer, retyper_matrix)
        retyped = torch.reshape(prev_layer, new_map_shape)

