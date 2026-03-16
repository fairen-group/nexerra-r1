# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Masks for training and inference (in order for implementing self- and cross-attention mechanisms)
# Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn

def generate_key_padding_mask(src: torch.LongTensor, pad_index: int = 0) -> torch.BoolTensor:
    '''Generate a boolean mask (batch, seq_len) where true indicates a padding token'''
    return (src == pad_index)

def generate_causal_mask(seq_len: int) -> torch.BoolTensor:
    '''Generate a causal mask (seq_len, seq_len) for the decoder.'''
    # --- standard upper triangular convention --- 
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal = 1).bool()
    return mask

# ---------------------------------
# Word dropout mask 
# I doubt this is used in production, but could come in handy if the decoder is too poweful
# ---------------------------------
class WordDropout(nn.Module):
    '''Standard implementation of the word dropout technique for token sequences;
    Randomly replaces a fraction of the tokens with an [UNK] token
    --- 
    Tried using this cause I initially found the transformer decoder too strong,
    but eventually got better results tuning beta
    '''
    def __init__(self, pad_token_id: int, unk_token_id: int):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
    
    def forward(self, dropout_prob: float, input_ids: torch.Tensor) -> torch.Tensor:
        '''only applies during model training and when dropout_prob > 0.0
        Example, dropout_prob = 0.1,
        Input: [SOS][TOK1]...[TOK10][PAD]...[EOS]
        Output: [SOS][TOK1]...[UNK]...[TOK10][PAD]...[EOS]'''

        if not self.training or dropout_prob == 0.0: return input_ids
        
        dropout_mask = (torch.rand_like(input_ids, dtype = torch.float) < dropout_prob) \
                        & (input_ids != self.pad_token_id)
        output_ids = input_ids.clone()
        output_ids[dropout_mask] = self.unk_token_id
        return output_ids 