# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Encoder block for the VAE. 
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
from torch.nn import TransformerEncoder as TransformerEncoder
from torch.nn import TransformerEncoderLayer as TransformerEncoderLayer

# --- Standard library imports ---
import math
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Positional encoding function ---
def PositionalEncoding(device, d_model: int, max_len: int) -> torch.Tensor:
    '''Standard sinusoidal positional encoding.
    Adapted from arXiv:1706.03762'''

    pe = torch.zeros(max_len, d_model, device = device)
    position = torch.arange(0, max_len, dtype = torch.float32, device = device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype = torch.float32, device = device) *
                        (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class TransEncoder(nn.Module):
    '''Standard transformer encoder adapted from arXiv:1706.03762;
    In case the sinusoidal positional encoding is not effective, RoPE may be adapted.
    Note: The shape_flag is used for debugging purposes. For production, set it to 0'''

    def __init__(self, device, shape_flag: int, vocab_size: int, d_model: int, num_head: int, num_layers: int, d_feedforward: int,
                 dropout: float, activation: str, max_len: int): 
        super().__init__()
        self.device = device
        self.shape_flag = shape_flag
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = Embedding(vocab_size, d_model)
        transformer_encoder_layer = TransformerEncoderLayer(d_model, num_head, d_feedforward, dropout, activation, batch_first = True)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers)
        pe = PositionalEncoding(self.device, self.d_model, self.max_len)
        self.register_buffer('pe', pe)

    def forward(self, src: torch.LongTensor, src_mask: torch.BoolTensor = None, src_key_padding_mask: torch.BoolTensor = None) -> torch.FloatTensor:
        src = self.token_embedding(src) * math.sqrt(self.d_model)
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Source shape: {src.size()}") # (batch, seq_len, d_model)
        # -----------------------------------------------------------------
        src += self.pe[: src.size(1), :].unsqueeze(0)
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Source shape after positional encoding: {src.size()}") # (batch, seq_len, d_model)
        # -----------------------------------------------------------------
        encoder_output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Encoder output shape: {encoder_output.size()}") # (batch, seq_len, d_model)
        # -----------------------------------------------------------------
        return encoder_output 
 
