#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:50:32 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

from params import *

#%%
'Positional Encoding'

class pos_encoder(nn.Module):
    
    def __init__(self, emb_size = emb_size, 
                 drop_rate: float = .1,
                 max_seq_len: int = seq_length, 
                 for_decoder_input: bool = False):
        
        super().__init__()
        
        self.emb_size = emb_size
        self.drop = nn.Dropout(drop_rate) if drop_rate != 0 else nn.Identity()
        self.for_decoder_input = for_decoder_input
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2
                                          ) * (-math.log(10000.0) / emb_size))
        pos_embedding = torch.zeros((max_seq_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.register_buffer('pos_embedding', pos_embedding)
    
    def forward(self, x: Tensor):
        
        if self.for_decoder_input:
            
            return self.pos_embedding[:x.size(1)
                                      ].repeat(x.size(0), 1, 1
                                               ).view(x.size(0), -1, 
                                                      self.emb_size)
            
        else:
            
            x = x + self.pos_embedding[:x.size(0)]
        
        return self.drop(x)

#%%
'Check'

if __name__ == '__main__':
    
    from cnn_embedding import *
    
    a = Variable(torch.randn(10, d_input[0], seq_length))
    m = nn.Sequential(cnn_embedding(d_input[0]),pos_encoder())
    a_out = m(a)
    print('a.shape: ',a.shape)
    print('a_out.shape: ',a_out.shape)
    
    a_out = m(a)
    print('a.shape: ',a.shape)
    print('a_out.shape: ',a_out.shape)
    
    a1 = Variable(torch.randn(10, 1, 50))
    m = nn.Sequential(cnn_embedding(d_input[1]),pos_encoder())
    a1_out = m(a1)
    print('a1.shape: ',a1.shape)
    print('a1_out.shape: ',a1_out.shape)
    
    a1_out = m(a1)
    print('a1.shape: ',a1.shape)
    print('a1_out.shape: ',a1_out.shape)
    
    a2 = Variable(torch.randn(10, d_input[0], seq_length))
    m = nn.Sequential(cnn_embedding(d_input[0]),
                      pos_encoder(for_decoder_input = True))
    a2_out = m(a2)
    print('a2.shape: ',a2.shape)
    print('a2_out.shape: ',a2_out.shape)
    
    a2_out = m(a2)
    print('a2.shape: ',a2.shape)
    print('a2_out.shape: ',a2_out.shape)
