#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:34:13 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

#%%
'Parameters and Hyper-Parameters'

d_input = (12, 1)
emb_size = 512
seq_length = 500
max_rr_seq = 30
batch_size = 32
model_out = 5
loss_fn = nn.BCEWithLogitsLoss()

#%%
"Setting GPU"

use_cpu = False
m_seed = 42

if use_cpu:
    device = torch.device('cpu')

elif torch.cuda.is_available(): 
    device = torch.device('cuda')
    torch.cuda.manual_seed(m_seed)
    torch.cuda.empty_cache()

elif not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was "
              "NOT built with MPS enabled.")
    else:
        print("MPS not available because this MacOS version is NOT 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")
print(device)