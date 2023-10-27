#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


class FrontTensorProduct(torch.nn.Module):
    def __init__(self, num_relation=4):
        super(FrontTensorProduct, self).__init__()
        self.num_relation = num_relation
        
    def forward(self, A, B):
        # relation dim in front
        if len(A.shape) == 3: # without batch
            n3, n1, n2 = A.shape
        elif len(A.shape) == 4: # with batch
            b, n3, n1, n2 = A.shape
            
        if len(B.shape) == 3:  
            n3, n2, n = B.shape
        elif len(B.shape) == 4: # with batch
            b, n3, n2, n = B.shape
        
        A_relation_index = 0 if len(A.shape) == 3 else 1
        B_relation_index = 0 if len(B.shape) == 3 else 1
        C_relation_index = A_relation_index | B_relation_index
            
        A_transformed = torch.fft.fft(input=A, dim=A_relation_index)
        B_transformed = torch.fft.fft(input=B, dim=B_relation_index)
        
        C_transformed = []
        
        for k in range(n3):
            a_slice = torch.squeeze(A_transformed.narrow(A_relation_index,k,1), dim=A_relation_index)
            b_slice = torch.squeeze(B_transformed.narrow(B_relation_index,k,1), dim=B_relation_index)
            C_transformed.append(torch.matmul(a_slice,b_slice))
        C = torch.fft.ifft(input = torch.stack(C_transformed, dim=C_relation_index), dim=C_relation_index)
        return torch.real(C)


# In[4]:


class SpatialAttentionBlock(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size=1, padding=0, norm_num_groups=1, norm_num_channel=1, affine=True):
        
        super(SpatialAttentionBlock, self).__init__()
        self.spatail_atten = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(norm_num_groups, inchannels, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(norm_num_groups, inchannels, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        atten_value = self.spatail_atten(x)
        atten_output = atten_value * x
        return atten_output


# In[5]:


class ChannelAttentionBlock(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelAttentionBlock, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, _, _ = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


# In[6]:

class TensorGraphSpatial2DConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, img_size, num_relation, bias=False, activation=None):
        super(TensorGraphSpatial2DConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.kernel_size_h = kernel_size
        self.kernel_size_w = kernel_size
        self.padding = padding
        self.activation = activation
        self.img_size = img_size
        self.num_relation = num_relation
        
        # Tensor Product
        self.tproduct = FrontTensorProduct(num_relation=num_relation)
        
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, 
                                                self.in_channels,
                                                self.kernel_size_h, 
                                                self.kernel_size_w,
                                               ))
        nn.init.xavier_normal_(self.weight)
        
        self.spatial_atten_block = SpatialAttentionBlock(inchannels=in_channels, outchannels=in_channels)
        self.channel_atten_block = ChannelAttentionBlock(num_channels=self.num_relation)

        self.b = nn.Parameter(torch.Tensor(self.in_channels, 1, 1))
        nn.init.xavier_normal_(self.b)
            
        
#     @torchsnooper.snoop()
    def forward(self, x, adj):
               
        b, c, h, w = x.shape
        x = self.spatial_atten_block(x)
        
        x = x.reshape(b, self.num_relation, h*w, c//self.num_relation)
        
        x = self.channel_atten_block(x)
        
        d = c//self.num_relation
        c = self.num_relation
        
#         print("adj, x:", adj.shape, x.shape)
       
        tensorgraph_constraint = self.tproduct(adj, x)
        
#         print("tensorgraph_constraint:", tensorgraph_constraint.shape)
        
        graph_constraint = tensorgraph_constraint
        
        graph_constraint = graph_constraint.reshape(-1, c*d, h, w)
        
#         print("tensorgraph_constraint:", tensorgraph_constraint.shape)
    
#         print("tensorgraph_constraint, b:", tensorgraph_constraint.shape, self.b.shape)
    
        if self.bias == True:
            graph_constraint += self.b
        
        graph_spatial_conv = F.conv2d(graph_constraint, self.weight,padding=self.padding,)
        
        if self.activation != None:
            graph_spatial_conv = self.activation(graph_spatial_conv)
        
        return torch.squeeze(graph_spatial_conv)


# In[7]:


class TLS_Convlstm_cell(nn.Module):
    def __init__(self, config):
        super().__init__()
        # hyperparrams
        self.input_dim = int(config['model']['in_channel'])
        self.input_channels = int(config['model']['hidden_dim'])
        self.hidden_dim = int(config['model']['hidden_dim'])
        self.kernel_size = int(config['model']['kernel_size'])
        self.padding = int(config['model']['padding'])
        self.img_size = (int(config['data']['img_height']), int(config['data']['img_width']))
        self.device = torch.device(str(config['model']['device']))
        self.conv2d = TensorGraphSpatial2DConvolution(self.input_channels + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              img_size = self.img_size,
                              num_relation = self.input_dim,
                              padding=self.padding)
        self.gn = nn.GroupNorm(4 * self.hidden_dim, 4 * self.hidden_dim)


    def forward(self, x, hidden, adj):
        h, c = hidden
        combined = torch.cat([x, h], dim=1)  # (batch_size, input_dim + hidden_dim, img_size[0], img_size[1])
        
        combined_conv = self.conv2d(combined, adj)  # (batch_size, 4 * hidden_dim, img_size[0], img_size[1])
        combined_conv = self.gn(combined_conv)
        i, f, o, g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        #Finish typical Convlstm above in the forward()
        #Attention below

        return h_next, (h_next, c_next)

    def init_hidden(self, batch_size, img_size):  # h, c, m initalize
        h, w = img_size
        
        return (torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device))

