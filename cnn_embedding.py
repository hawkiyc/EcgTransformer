#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:43:10 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

from params import *

#%%
'CNN Embedding'

class cnn_embedding(nn.Module):
    
    def __init__(self, in_dim: int, emb_dim = emb_size, kernel_size = 3, 
                 batch_norm = False, act_fun: nn.Module = nn.ReLU()):
        
        assert kernel_size % 2 == 1, 'Kernel size shall be odd number'
        super(cnn_embedding, self).__init__()
        
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.act_fun = act_fun
        self.pool = nn.MaxPool1d(kernel_size = kernel_size, stride = 1)
        
        self.block_0 = self.conv_block(in_dim, emb_dim)
        self.block_1 = self.conv_block(emb_dim, emb_dim)
        # self.block_2 = self.conv_block(int(emb_dim/4), int(emb_dim/2))
        # self.block_3 = self.conv_block(int(emb_dim/2), emb_dim)
        
    def conv_block(self, in_, out_):
        
        sub = [nn.Conv1d(in_, out_, 
                         self.kernel_size, 
                         padding = 'same', 
                         bias = False if self.batch_norm else True),
               nn.BatchNorm1d(out_) if self.batch_norm else nn.Identity(),
               self.act_fun]
        
        return nn.Sequential(*sub)
    
    def forward(self, x):
        
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_1(x)
        x = self.pool(x)
        x = self.block_1(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        
        return x

#%%
'Check'

if __name__ == '__main__':
    
    a = Variable(torch.randn(10, d_input[0], seq_length))
    m = cnn_embedding(d_input[0])
    a_out = m(a)
    print(a.shape)
    print(a_out.shape)