# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# The VAE bottleneck. 
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

# --- Standard library imports ---
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VAEBottleneck(nn.Module):
    '''Standard implementation of the VAE bottleneck arXiv:1312.6114v11'''
    def __init__(self, shape_flag: int, d_model: int = 512, latent_dim: int = 128):
        super().__init__()
        self.shape_flag = shape_flag
        self.pooling = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)

    def forward(self, encoder_output: torch.FloatTensor, src_key_padding_mask: torch.BoolTensor = None):
        if src_key_padding_mask is not None:
            pooling_mask = ~src_key_padding_mask 
            pooling_mask = pooling_mask.float().unsqueeze(-1)
            pooled_output = (encoder_output * pooling_mask).sum(dim = 1) / pooling_mask.sum(dim = 1)
            # -----------------------------------------------------------------
            if self.shape_flag == 1: print(f"Pooled output shape (with src_key_padding_mask): {pooled_output.size()}") # (batch, d_model)
            # -----------------------------------------------------------------
        else:
            pooled_output = encoder_output.mean(dim = 1) 
            # -----------------------------------------------------------------
            if self.shape_flag == 1: print(f"Pooled output shape (without src_key_padding_mask): {pooled_output.size()}") # (batch, d_model)
            # -----------------------------------------------------------------
        pooled_output = self.pooling(pooled_output) 
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Pooled output shape after pooling: {pooled_output.size()}") # (batch, d_model)
        # -----------------------------------------------------------------
        mu = self.mu(pooled_output) 
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Mu shape: {mu.size()}") # (batch, latent_dim)
        # -----------------------------------------------------------------
        logvar = self.logvar(pooled_output)
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Logvar shape: {logvar.size()}") # (batch, latent_dim) 
        # -----------------------------------------------------------------
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent_variable = mu + (eps * std) # standard reparam trick
        # -----------------------------------------------------------------
        if self.shape_flag == 1: print(f"Latent variable shape: {latent_variable.size()}") # (batch, latent_dim)
        # -----------------------------------------------------------------
        return latent_variable, mu, logvar