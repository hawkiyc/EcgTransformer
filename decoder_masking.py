# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:04:47 2023

@author: Revlis_user
"""

#%%
'Import Library'

from params import *

#%%
'Masking'

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

#%%
'Masking'

tgt_ecg_mask = generate_square_subsequent_mask(dim1=seq_length - 4,
                                               dim2=seq_length - 4)
tgt_rr_mask = generate_square_subsequent_mask(dim1=max_rr_seq - 4,
                                              dim2=max_rr_seq - 4)

src_ecg_mask = generate_square_subsequent_mask(dim1=seq_length - 4,
                                               dim2=seq_length - 4)
src_rr_mask = generate_square_subsequent_mask(dim1=max_rr_seq - 4, 
                                              dim2=max_rr_seq - 4)