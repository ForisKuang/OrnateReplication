import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np

class OrnateReplicaModel(nn.Module):

    def __init__(self, num_retype=15):
        super(OrnateReplicaModel, self).__init__()
        self.num_retype = num_retype
        self.activation = nn.ELU()
        self.final_activation = nn.Sigmoid()
        self.CONV1 = 20
        self.CONV2 = 30
        self.CONV3 = 20
        self.NB_TYPE = 167
        self.NB_DIMOUT = 4*4*4*self.CONV3
        self.batchNorm = nn.BatchNorm3d(167*24*24*24)
        self.batchNorm1d = nn.BatchNorm1d(self.NB_DIMOUT)
        self.apply_conv1 = self.conv(self.num_retype, self.CONV1, 3)
        self.apply_conv2 = self.conv(self.CONV1, self.CONV2, 4)
        self.apply_conv3 = self.conv(self.CONV2, self.CONV3, 3)
        # Default dropout is set to 0.5 which is the same as Ornate
        self.dropout = nn.Dropout()
        self.avgpool3d = nn.AvgPool3d(4, stride=4)

    def conv(self, in_dim, out_dim, kernel_size, stride=1):
        return nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, padding=0,
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

        # Apply first convolution of kernel size 3
        # Can experiment with bigger strides
        prev_layer = self.apply_conv1(retyped)

        # Apply batch normalization, with num_features
        prev_layer = self.batchNorm(prev_layer)
        
        # Apply dropout to prevent overfitting
        prev_layer = self.dropout(prev_layer)

        # Apply activation function
        prev_layer = self.activation(prev_layer)
        
        # Apply second convolution with kernel size 4
        prev_layer = self.apply_conv2(prev_layer)
        
        # Apply batch normalization, with num_features
        prev_layer = self.batchNorm(prev_layer)

        # Apply activation function
        prev_layer = self.activation(prev_layer)

        # Apply second convolution with kernel size 4
        prev_layer = self.apply_conv3(prev_layer)
        
        # Apply batch normalization, with num_features
        prev_layer = self.batchNorm(prev_layer)

        # Apply activation function
        prev_layer = self.activation(prev_layer)

        prev_layer = self.avgpool3d(prev_layer)
        
        NB_DIMOUT = self.CONV3 * 4 * 4 * 4

        flat0 = torch.reshape(prev_layer, (-1, NB_DIMOUT))
        
        prev_layer = self.batchNorm1d(flat0)
        
        flat1 = self.activation(prev_layer)
        
        prev_layer = torch.squeeze(flat1)
        
        return self.final_activation(prev_layer)
