#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:43:15 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

from params import *

#%%
'Self Attention Pooling'

class SelfAttentionPooling(nn.Module):
    """
    
    Implementation of SelfAttentionPooling Original Paper: 
        Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    
    Source Code from 'pohanchi' on Github:
    https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, 
            H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1), dim =1 ).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        utter_rep = utter_rep.unsqueeze(1)

        return utter_rep

#%%
'Check'

if __name__ == '__main__':
    
    from cnn_embedding import *
    from pos_encoder import *
    
    a = Variable(torch.randn(10, d_input[0], seq_length))
    m = nn.Sequential(cnn_embedding(d_input[0]),
                      pos_encoder(),
                      SelfAttentionPooling(input_dim = emb_size))
    a_out = m(a)
    print('a.shape: ',a.shape)
    print('a_out.shape: ',a_out.shape)
    
    a1 = Variable(torch.randn(10, 1, 50))
    m = nn.Sequential(cnn_embedding(d_input[1]),
                      pos_encoder(),
                      SelfAttentionPooling(input_dim = emb_size))
    a1_out = m(a1)
    print('a1.shape: ',a1.shape)
    print('a1_out.shape: ',a1_out.shape)
    