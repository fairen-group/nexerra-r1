# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Complete TVAE compilation. 
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
import torch.nn.functional as F

# --- Local imports ---
from .Encoder import TransEncoder 
from .VAEBottleneck import VAEBottleneck
from .Decoder import TransDecoder
from nexerra.utils.masks import generate_causal_mask

# --- Standard library imports ---
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    '''Zero out all but the top p cumulative probability mass of the logits'''
    sorted_logits, sorted_indices = torch.sort(logits, descending = True, dim = -1)
    sorted_probs = F.softmax(sorted_logits, dim = -1)
    cum_probs = torch.cumsum(sorted_probs, dim = -1)
    mask = cum_probs > p
    mask[..., 0] = False
    mask_original = torch.zeros_like(mask).scatter_(1, sorted_indices, mask)
    filtered_logits = logits.masked_fill(mask_original, float('-inf'))
    return filtered_logits

class VAEModel(nn.Module):
    '''Complete Transformer VAE model compilation'''
    def __init__(self, device, shape_flag: int, vocab_size: int, sos_index: int, eos_index: int, pad_index: int, unk_index: int, 
                 d_model: int, latent_dim: int, num_head: int, num_encoder_layers: int, num_decoder_layers: int, 
                 d_feedforward: int,  max_len: int, activation: str, encoder_dropout: float, decoder_dropout: float):
        super().__init__()
        self.device = device
        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.unk_index = unk_index 
        self.d_model = d_model

        self.encoder = TransEncoder(device, shape_flag, vocab_size, d_model, num_head, num_encoder_layers, d_feedforward,
                                        encoder_dropout, activation, max_len)
        self.bottleneck = VAEBottleneck(shape_flag, d_model, latent_dim)
        self.decoder = TransDecoder(device, shape_flag, vocab_size, d_model, latent_dim, num_head, num_decoder_layers, 
                                        d_feedforward, decoder_dropout, max_len)
        
    def forward(self, src: torch.LongTensor, tgt: torch.LongTensor, src_mask: torch.BoolTensor = None, 
                src_key_padding_mask: torch.BoolTensor = None, tgt_mask: torch.BoolTensor = None, 
                tgt_key_padding_mask: torch.BoolTensor = None):
        
        encoder_output = self.encoder(src, src_mask, src_key_padding_mask)
        latent_variable, mu, logvar = self.bottleneck(encoder_output, src_key_padding_mask)
        decoder_output = self.decoder(tgt, latent_variable, tgt_mask, tgt_key_padding_mask)
        return decoder_output, latent_variable, mu, logvar
    
    def encode(self, src: torch.LongTensor, src_mask: torch.BoolTensor = None, src_key_padding_mask: torch.BoolTensor = None):
        encoder_output = self.encoder(src, src_mask, src_key_padding_mask)
        latent_variable, mu, logvar = self.bottleneck(encoder_output, src_key_padding_mask)
        return latent_variable, mu, logvar
    
    def decode(self, tgt: torch.LongTensor, latent_variable: torch.Tensor, tgt_mask: torch.BoolTensor = None, 
               tgt_key_padding_mask: torch.BoolTensor = None) -> torch.FloatTensor:
        decoder_output = self.decoder(tgt, latent_variable, tgt_mask, tgt_key_padding_mask)
        return decoder_output
    
    def generate(self, latent_variable: torch.Tensor, max_len: int, temperature: float = 0.9, top_p_val: float = 0.9) -> torch.LongTensor:
        batch_size = latent_variable.size(0)
        device = latent_variable.device
        sos = self.sos_index
        generated = torch.full((batch_size, 1), sos, device = device, dtype = torch.long)
        finished = torch.zeros(batch_size, dtype = torch.bool, device = device)
        
        for _ in range(max_len):
            target_mask = generate_causal_mask(generated.size(1)).to(device)
            output = self.decode(tgt = generated, latent_variable = latent_variable, 
                                 tgt_mask = target_mask, tgt_key_padding_mask = None)
            logits = output[:, -1, :] 
            logits = logits / temperature # temperature controls the smoothness/sharpness of the distribution

            logits[:, self.pad_index] = float('-inf')
            logits = top_p(logits, top_p_val) if top_p_val > 0.0 else logits
            probs = F.softmax(logits, dim = -1)
            token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, token], dim = 1)
            finished |= (token.squeeze(-1) == self.eos_index)
            if finished.all(): break
        
        return generated
    
    
