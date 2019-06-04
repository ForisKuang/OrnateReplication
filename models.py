import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np

class Discriminator(nn.Module):

    def __init__(self, num_retype=15, device='cpu'):
        super(Discriminator, self).__init__()
        self.num_retype = num_retype
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()
        self.CONV1 = 20
        self.CONV2 = 30
        self.CONV3 = 20
        self.NB_TYPE = 167
        self.NB_DIMOUT = 4*4*4*self.CONV3
        
        
        # TODO: Page 3 of the 3D-IWGAN paper suggests not using Batch norm for discriminator.
        # But it's present in the Pytorch GAN tutorial...
        self.batchNorm1 = nn.BatchNorm3d(self.CONV1)
        self.batchNorm2 = nn.BatchNorm3d(self.CONV2)
        self.batchNorm1d = nn.BatchNorm1d(self.NB_DIMOUT)
        self.apply_conv1 = self.conv(self.num_retype, self.CONV1, 3)
        self.apply_conv2 = self.conv(self.CONV1, self.CONV2, 4)
        self.apply_conv3 = self.conv(self.CONV2, self.CONV3, 3)
        self.lin1 = nn.Linear(self.NB_DIMOUT, 512)
        self.lin2 = nn.Linear(512, 200)
        self.lin3 = nn.Linear(200, 1)
        # Default dropout is set to 0.5 which is the same as Ornate
        self.dropout = nn.Dropout()
        self.avgpool3d = nn.AvgPool3d(4, stride=4)
        self.device = device

    def conv(self, in_dim, out_dim, kernel_size, stride=1):
        return nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, padding=0,
                stride=stride, bias=False)

    def forward(self, features):

        # Retyper: a [high dimension * low dimension] tensor
        retyper_matrix = torch.nn.Parameter(torch.rand(self.NB_TYPE, self.num_retype), requires_grad=True).to(self.device)

        # (batch_size, 24, 24, 24, 167)
        shape = features.shape

        # Reshape so that we have a two-dimensional tensor.
        # Each row will represent an (x,y,z) point, which has a 167-dimensional feature vector.
        prev_layer = torch.reshape(features, (-1, self.NB_TYPE))

        # Multiply each (x,y,z) point's feature vector by the retyper matrix,
        # to reduce the dimensionality of the feature vectors
        # (batch_size x 24 x 24 x 24, 167)
        prev_layer = torch.mm(prev_layer, retyper_matrix)

        # (batch_size x 24 x 24 x 24, 15)
        retyped = torch.reshape(prev_layer, (shape[0], shape[1], shape[2], shape[3], self.num_retype))

        # (batch_size, 24, 24, 24, 15)
        retyped = retyped.permute(0, 4, 1, 2, 3)

        # Apply first convolution of kernel size 3
        # Can experiment with bigger strides
        # (batch_size, 15, 24, 24, 24)
        prev_layer = self.apply_conv1(retyped)

        # Apply batch normalization, with num_features
        prev_layer = self.batchNorm1(prev_layer)
        
        # Apply dropout to prevent overfitting
        prev_layer = self.dropout(prev_layer)

        # Apply activation function
        prev_layer = self.activation(prev_layer)
        
        # Apply second convolution with kernel size 4
        prev_layer = self.apply_conv2(prev_layer)
        
        # Apply batch normalization, with num_features
        prev_layer = self.batchNorm2(prev_layer)

        # Apply activation function
        prev_layer = self.activation(prev_layer)

        # Apply second convolution with kernel size 4
        prev_layer = self.apply_conv3(prev_layer)
        
        # Apply batch normalization, with num_features
        prev_layer = self.batchNorm1(prev_layer)

        # Apply activation function
        prev_layer = self.activation(prev_layer)

        prev_layer = self.avgpool3d(prev_layer)

        prev_layer = prev_layer.reshape(prev_layer.size()[0], -1)
        prev_layer = self.lin1(prev_layer)
        prev_layer = F.relu(prev_layer)

        prev_layer = self.lin2(prev_layer)
        prev_layer = F.relu(prev_layer)
        prev_layer = self.lin3(prev_layer)

        # Apply sigmoid at the end
        prev_layer = self.final_activation(prev_layer)
        return prev_layer

class SurfaceVAE(nn.Module):

    def __init__(self, num_retype=15, device='cpu'):
        super(SurfaceVAE, self).__init__()
        self.num_retype = num_retype
        self.activation = nn.LeakyReLU()
        #self.means_activation = nn.Identity()
        self.sigmas_activation = nn.Tanh()
        self.CONV1 = 20
        self.CONV2 = 30
        self.CONV3 = 20
        self.CONV4 = 20
        self.NB_TYPE = 167
        self.batchNorm1 = nn.BatchNorm3d(self.CONV2)
        self.batchNorm2 = nn.BatchNorm3d(self.CONV3)
        self.batchNorm3 = nn.BatchNorm3d(self.CONV4)

        self.apply_conv1 = self.conv(self.num_retype, self.CONV1, 3, stride=1, padding=1)
        self.apply_conv2 = self.conv(self.CONV1, self.CONV2, 4, stride=2, padding=1)
        self.apply_conv3 = self.conv(self.CONV2, self.CONV3, 4, stride=2, padding=1)
        self.apply_conv4 = self.conv(self.CONV3, self.CONV4, 4, stride=2, padding=1)

        self.means_linear = nn.Linear(20*3*3*3, 400)
        self.sigmas_linear = nn.Linear(20*3*3*3, 400)

        # Default dropout is set to 0.5
        self.dropout = nn.Dropout()
        self.device = device

    def conv(self, in_dim, out_dim, kernel_size, stride=1, padding=0):
        return nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, padding=padding,
                stride=stride, bias=False)

    def forward(self, features):

        # Retyper: a [high dimension * low dimension] tensor
        retyper_matrix = torch.nn.Parameter(torch.rand(self.NB_TYPE, self.num_retype), requires_grad=True).to(self.device)

        # (batch_size, 24, 24, 24, 167)
        shape = features.shape

        # Reshape so that we have a two-dimensional tensor.
        # Each row will represent an (x,y,z) point, which has a 167-dimensional feature vector.
        prev_layer = torch.reshape(features, (-1, self.NB_TYPE)).to(self.device)

        # Multiply each (x,y,z) point's feature vector by the retyper matrix,
        # to reduce the dimensionality of the feature vectors
        # (batch_size x 24 x 24 x 24, 167)
        prev_layer = torch.mm(prev_layer, retyper_matrix)

        # (batch_size x 24 x 24 x 24, 15)
        retyped = torch.reshape(prev_layer, (shape[0], shape[1], shape[2], shape[3], self.num_retype))

        # (batch_size, 24, 24, 24, 15)
        retyped = retyped.permute(0, 4, 1, 2, 3)

        # Apply first convolution of kernel size 4
        # Stride of 2 to begin with
        # (batch_size, 15, 24, 24, 24)
        prev_layer = self.apply_conv1(retyped)
    
        # Apply Leaky ReLU to first convolution
        prev_layer = self.activation(prev_layer)

        # Apply conv2
        prev_layer = self.apply_conv2(prev_layer)

        # Apply batch normalization, with num_features
        prev_layer = self.batchNorm1(prev_layer)
        
        # Apply dropout to prevent overfitting
        prev_layer = self.dropout(prev_layer)

        # Apply activation function
        prev_layer = self.activation(prev_layer)
        
        # Apply second convolution with kernel size 4
        prev_layer = self.apply_conv3(prev_layer)
        
        # Apply batch normalization, with num_features
        prev_layer = self.batchNorm2(prev_layer)

        # Apply activation function
        prev_layer = self.activation(prev_layer)

        # Apply second convolution with kernel size 4
        prev_layer = self.apply_conv4(prev_layer)
        
        # Apply batch normalization, with num_features
        prev_layer = self.batchNorm3(prev_layer)

        # Flatten layer with respect to Batch Size
        prev_layer = prev_layer.reshape(prev_layer.size()[0], -1)

        means = self.means_linear(prev_layer)
        #means = self.means_activation(means)

        sigmas = self.sigmas_linear(prev_layer)
        sigmas = self.sigmas_activation(sigmas)

        return means, sigmas

class Generator(nn.Module):

    def __init__(self, num_retype=15):
        super(Generator, self).__init__()
        self.num_retype = num_retype
        self.activation = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.DECONV0 = 20
        self.DECONV1 = 40
        self.DECONV2 = 80
        self.DECONV3 = 167
        self.linear1 = nn.Linear(400, 20*3*3*3)
        self.batchNorm1 = nn.BatchNorm3d(self.DECONV0)
        self.batchNorm2 = nn.BatchNorm3d(self.DECONV1)
        self.batchNorm3 = nn.BatchNorm3d(self.DECONV2)
        self.apply_deconv1 = self.deconv(self.DECONV0, self.DECONV1, 4, stride=2)  # [batch size * 3 * 3 * 3 * 20] --> [batch size * 6 * 6 * 6 * 40]
        self.apply_deconv2 = self.deconv(self.DECONV1, self.DECONV2, 4, stride=2)  # [batch size * 6 * 6 * 6 * 40] --> [batch size * 12 * 12 * 12 * 80]
        self.apply_deconv3 = self.deconv(self.DECONV2, self.DECONV3, 4, stride=2)  # [batch size * 12 * 12 * 12 * 80] --> [batch size * 24 * 24 * 24 * 167]

        # Default dropout is set to 0.5
        self.dropout = nn.Dropout()

    def deconv(self, in_dim, out_dim, kernel_size, stride=1):
        return nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size, padding=0,
                stride=stride, bias=False)

    def forward(self, features):
        
        start_layer = self.linear1(features)
        
        reshape_layer = torch.reshape(start_layer, (-1, 3, 3, 3, 20))

        prev_layer = self.batchNorm1(reshape_layer)

        prev_layer = self.activation(prev_layer)
 
        prev_layer = self.apply_deconv1(prev_layer)
        prev_layer = self.batchNorm2(prev_layer)
        prev_layer = self.activation(prev_layer)

        prev_layer = self.apply_deconv2(prev_layer)
        prev_layer = self.batchNorm3(prev_layer)
        prev_layer = self.activation(prev_layer)

        prev_layer = self.apply_deconv3(prev_layer)
        prev_layer = torch.reshape(prev_layer, (-1, 24, 24, 24, 167))

        final_activation = self.tanh(prev_layer)

        return final_activation


