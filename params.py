#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:34:13 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

import ast
from datetime import datetime
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer
from tslearn.preprocessing import TimeSeriesResampler as tsr
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import wfdb

#%%
'Parameters and Hyper-Parameters'

path = '../../DATA/PTB-XL'
sr = 100
add_noise = False
check_data = True
scaler = False
LEADs = [ "DI", "DII", "DIII", "AVL", "AVR", "AVF", 
         "V1", "V2", "V3", "V4", "V5", "V6"]
diagnose_label = ['CD', 'HYP', 'MI', "NORM", 'STTC']
model_out = len(diagnose_label)
d_input = (12, 1)
emb_size = 512
seq_length = 500
max_rr_seq = 20
batch_size = 24
n_epochs = 50
loss_fn = nn.BCEWithLogitsLoss()
out_activation = nn.Sigmoid()

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

#%%
'Make Output Dir'

if not os.path.isdir('results'):
    os.makedirs('results')
