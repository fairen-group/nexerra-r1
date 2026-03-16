# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Decoder block for the VAE. 
# Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Pytorch imports ---
import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn import TransformerDecoder as TransformerDecoder
from torch.nn import TransformerDecoderLayer as TransformerDecoderLayer

from nexerra.model.Encoder import PositionalEncoding

# --- Standard library imports ---
import math
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransDecoder(nn.Module):
    '''Standard transformer decoder adapted from arXiv:1706.03762'''
    def __init__(self, device, shape_flag: int, vocab_size: int, d_model: int = 512, latent_dim: int = 64, num_head: int = 8, num_layers: int = 6, 
                 d_feedforward: int = 2048, dropout: float = 0.1, max_len: int = 114):
        super().__init__()
        self.device = device
        self.shape_flag = shape_flag
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = Embedding(vocab_size, d_model)
        self.latent_projection = nn.Linear(latent_dim, d_model)
        self.transformer_decoder_layer = TransformerDecoderLayer(d_model, num_head, d_feedforward, dropout, batch_first = True)
        self.transformer_decoder = TransformerDecoder(self.transformer_decoder_layer, num_layers)
        self.decoder_projection = nn.Linear(d_model, vocab_size)
        pe = PositionalEncoding(self.device, self.d_model, max_len = self.max_len)
        self.register_buffer('pe', pe)
    
    def forward(self, tgt: torch.LongTensor, latent_variable: torch.Tensor, tgt_mask: torch.BoolTensor, tgt_key_padding_mask: torch.BoolTensor = None) -> torch.FloatTensor:
        tgt = self.token_embedding(tgt) * math.sqrt(self.d_model)
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Target shape: {tgt.size()}") # (batch, seq_len - 1, d_model)
        # -----------------------------------------------------------------
        tgt += self.pe[: tgt.size(1), : ].unsqueeze(0)
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Target shape after positional encoding: {tgt.size()}") # (batch, seq_len - 1, d_model)
        # -----------------------------------------------------------------
        batch_size = latent_variable.size(0)
        seq_len = tgt.size(1)
        latent_variable_projection = self.latent_projection(latent_variable).unsqueeze(1)
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Latent variable projection shape: {latent_variable_projection.size()}") # (batch, 1, d_model)
        # -----------------------------------------------------------------
        latent_variable_expanded = latent_variable_projection.expand(batch_size, seq_len, self.d_model)
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Latent variable expanded shape: {latent_variable_expanded.size()}") # (batch, seq_len - 1, d_model)
        # -----------------------------------------------------------------
        decoder_output = self.transformer_decoder(tgt = tgt, memory = latent_variable_expanded, tgt_mask = tgt_mask,
                                                 tgt_key_padding_mask = tgt_key_padding_mask)
        # -----------------------------------------------------------------
        if self.shape_flag == 1:  print(f"Decoder output shape before projection: {decoder_output.size()}") # (batch, seq_len - 1, d_model)
        # -----------------------------------------------------------------
        decoder_output = self.decoder_projection(decoder_output) 
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Decoder output shape after projection: {decoder_output.size()}") # (batch, seq_len - 1, vocab_size)
        # -----------------------------------------------------------------
        return decoder_output
    
