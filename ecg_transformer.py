#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:49:17 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

from params import *
from cnn_embedding import *
from pos_encoder import *
from self_attention_pooling import *
from decoder_masking import *

#%%
'TS-Transformer'

class ECGTransformer(nn.Module):
    
    def __init__(self, d_input: tuple = d_input, emb_dim: int = emb_size, 
                 emb_k_size: int = 3, emb_norm: bool = False,
                 max_seq_len: int = seq_length - 4, n_encoder_layers: int = 4, 
                 n_decoder_layers: int = 4, n_heads: int = 8, 
                 dropout_encoder: float = .1, dropout_decoder: float = .1, 
                 dropout_pos_encoder: float = .1, dim_feedforward: int = 2048,
                 use_self_att_pool: bool = True, fc_drop: int = .2, 
                 out_features: int = model_out, act: nn.Module = nn.ReLU()):
        
        super().__init__()
        
        'embedding layer'
        ecg_embedding_layer = cnn_embedding(d_input[0],
                                            emb_dim = emb_dim,
                                            kernel_size = emb_k_size,
                                            batch_norm = emb_norm,
                                            act_fun = act)
        rr_embedding_layer = cnn_embedding(d_input[1], 
                                           emb_dim = emb_dim,
                                           kernel_size = emb_k_size,
                                           batch_norm = emb_norm,
                                           act_fun = act)
        
        'positional embedding'
        encoder_pos_emb = pos_encoder(emb_size = emb_dim, 
                                      drop_rate = dropout_pos_encoder,
                                      max_seq_len = max_seq_len,)
        decoder_pos_emb = pos_encoder(emb_size = emb_dim, 
                                      drop_rate = dropout_pos_encoder,
                                      max_seq_len = max_seq_len, 
                                      for_decoder_input = True)
        
        'transformer encoder input'
        self.ecg_encoder_emb = nn.Sequential(ecg_embedding_layer, 
                                             encoder_pos_emb)
        self.rr_encoder_emb = nn.Sequential(rr_embedding_layer, 
                                            encoder_pos_emb)
        
        'transformer decoder input'
        self.ecg_decoder_emb = nn.Sequential(ecg_embedding_layer, 
                                             decoder_pos_emb)
        self.rr_decoder_emb = nn.Sequential(rr_embedding_layer, 
                                            decoder_pos_emb)
        
        'layer norm'
        norm = nn.LayerNorm(emb_dim)
        
        'transformer encoder'
        encoder_layer = nn.TransformerEncoderLayer(d_model = emb_dim, 
                                                   nhead = n_heads, 
                                                   dim_feedforward = 
                                                       dim_feedforward,
                                                   dropout = dropout_encoder,
                                                   activation = act,
                                                   batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 
                                             num_layers = n_encoder_layers, 
                                             norm = norm)
        
        'self attention pooling'
        self.self_att_pool = SelfAttentionPooling(input_dim = emb_dim
                                  ) if use_self_att_pool else nn.Identity()
        self.use_self_att_pool = use_self_att_pool
        
        'transformer decoder'
        decoder_layer = nn.TransformerDecoderLayer(d_model = emb_dim, 
                                                   nhead = n_heads, 
                                                   dim_feedforward = 
                                                       dim_feedforward,
                                                   dropout = dropout_decoder,
                                                   activation = act,
                                                   batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 
                                             num_layers = n_decoder_layers, 
                                             norm = norm)
        
        'FC'
        self.flatten = nn.Flatten()
        self.fc_in = self.get_fc_in()
        self.fc = nn.Sequential(nn.Linear(self.fc_in, 512),
                                act,
                                nn.Dropout(fc_drop),
                                nn.Linear(512, out_features))
    
    def get_fc_in(self, tgt_ecg_mask = tgt_ecg_mask, 
                  tgt_rr_mask = tgt_rr_mask, 
                  memory_ecg_mask = src_ecg_mask, 
                  memory_rr_mask = src_rr_mask,):
        
        'input/src embedding'
        pseudo_src0 = self.ecg_encoder_emb(Variable(
            torch.ones(2,d_input[0], seq_length)))
        pseudo_src1 = self.rr_encoder_emb(Variable(
            torch.ones(2,d_input[1], max_rr_seq)))
        
        'get encoder output'
        
        pseudo_src0 = self.encoder(pseudo_src0)
        pseudo_src0 = self.self_att_pool(pseudo_src0)
        pseudo_src1 = self.encoder(pseudo_src1)
        pseudo_src1 = self.self_att_pool(pseudo_src1)
        
        'tgt/positional embedding'
        pseudo_tgt0 = self.ecg_decoder_emb(Variable(
            torch.ones(2,d_input[0], seq_length)))
        pseudo_tgt1 = self.rr_decoder_emb(Variable(
            torch.ones(2,d_input[1], max_rr_seq)))
        
        'get decoder output'
        pseudo_output0 = self.decoder(
            tgt = pseudo_tgt0, 
            memory = pseudo_src0,
            tgt_mask = tgt_ecg_mask,
            memory_mask = None if self.use_self_att_pool else memory_ecg_mask)
        pseudo_output1 = self.decoder(
            tgt = pseudo_tgt1, 
            memory = pseudo_src1,
            tgt_mask = tgt_rr_mask, 
            memory_mask = None if self.use_self_att_pool else memory_rr_mask)
        
        pseudo_outputs = torch.cat([pseudo_output0, pseudo_output1], dim = 1)
        pseudo_outputs = self.flatten(pseudo_outputs)
        
        return pseudo_outputs.data.view(2, -1).size(1)
    
    def forward(self, ecg, rr, 
                tgt_ecg_mask = tgt_ecg_mask, 
                tgt_rr_mask = tgt_rr_mask, 
                memory_ecg_mask = src_ecg_mask, 
                memory_rr_mask = src_rr_mask,):
        
        'input/src embedding'
        ecg_src = self.ecg_encoder_emb(ecg)
        rr_src = self.rr_encoder_emb(rr)
        
        'get encoder output'
        ecg_src = self.encoder(src = ecg_src)
        ecg_src = self.self_att_pool(ecg_src)
        
        rr_src = self.encoder(src = rr_src)
        rr_src = self.self_att_pool(rr_src)
        
        'tgt/positional embedding'
        ecg_tgt = self.ecg_decoder_emb(ecg)
        rr_tgt = self.rr_decoder_emb(rr)
        
        'get decoder output'
        ecg_decoder_output = self.decoder(
            tgt = ecg_tgt,
            memory = ecg_src,
            tgt_mask = tgt_ecg_mask.to(device, torch.float32),
            memory_mask = None if self.use_self_att_pool 
                else memory_ecg_mask.to(device, torch.float32))
        rr_decoder_output = self.decoder(
            tgt = rr_tgt, 
            memory = rr_src, 
            tgt_mask = tgt_rr_mask.to(device, torch.float32), 
            memory_mask = None if self.use_self_att_pool 
                else  memory_rr_mask.to(device, torch.float32))
        
        outputs = torch.cat([ecg_decoder_output, rr_decoder_output], dim = 1)
        outputs = self.flatten(outputs)
        outputs = self.fc(outputs)
        
        return outputs

#%%
"Check"

if __name__ == '__main__':
    
    in_0 = Variable(torch.randn(batch_size * 2, d_input[0], seq_length))
    in_1 = Variable(torch.randn(batch_size * 2, d_input[1], max_rr_seq))
    
    out = Variable(torch.randn(batch_size * 2, model_out))
    
    pseudo_set = TensorDataset(in_0, in_1, out)
    pseudo_loader = DataLoader(dataset = pseudo_set,
                               batch_size = batch_size,
                               shuffle = False)
    
    model = ECGTransformer(use_self_att_pool=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x0,x1,y in pseudo_loader:
            x0,x1,y = x0.to(device), x1.to(device), y.to(device)
            pseudo_y_hat = model(x0,x1)
            pseudo_loss = loss_fn(pseudo_y_hat, y)
            print(pseudo_loss.item())