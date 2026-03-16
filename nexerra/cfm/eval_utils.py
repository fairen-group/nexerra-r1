# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  Modularised version of the evaluation of the OT-CFM for easier downstream processing.
#  Author: Dhruv Menon (dm958[at]cam[dot]ac[dot]uk)
#
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from torchdiffeq import odeint

# --- dopri (RK45) integrator ---
def rk45_integrator(
    vfield, # vector field that is integrated
    z0: torch.Tensor, 
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_steps: int = 10000,
):
    '''Integrate dz/dt = vfield(t, z) from t = 0 to 1 with adaptive RK45'''
    t = torch.tensor([0.0, 1.0], device = z0.device, dtype = z0.dtype)
    zt = odeint(
        func = lambda ts, zs: vfield(ts, zs),
        y0 = z0,
        t = t,
        method = 'dopri5',
        rtol = rtol,
        atol = atol,
        options = {'max_num_steps': max_steps}
    )
    return zt[-1]

# --- cfg vector field ---
def make_vfield(flow: nn.Module, 
                c: torch.Tensor, 
                guidance_scale: float = 0.0, 
                uncond_value: float = 0.0, 
                variant: str = 'standard'):
    def _expand(t, z, c_in):
        if t.dim() == 0:
            t = torch.full((z.size(0), 1), float(t), device = z.device, dtype = z.dtype)
        elif t.dim() == 1:
            t = t.view(-1, 1)
        if c_in.dim() == 1:
            c_in = c_in.view(-1, 1)
        if c_in.size(0) == 1 and z.size(0) > 1:
            c_in = c_in.expand(z.size(0), c_in.size(1))
        return t, c_in
    
    '''vector-field with classifier-free guidance'''
    if guidance_scale is None or guidance_scale <= 0.0:
        def vfield(t, z): 
            t, c_in = _expand(t, z, c)
            return flow(t, z, c_in)
        return vfield
    else:
        c0 = torch.full_like(c, float(uncond_value))
        def vfield(t, z):
            t, c_in = _expand(t, z, c)
            t, c0_in = _expand(t, z, c0)
            vc = flow(t, z, c_in)
            vu = flow(t, z, c0_in)
            if variant == 'conditional': return vc + guidance_scale * (vc - vu)
            return vu + guidance_scale * (vc - vu)
        return vfield

# --- generate SMILES from the flow + VAE ---
@torch.no_grad()
def generate_smiles(
    flow: nn.Module,
    vae,
    tokenizer,
    n_samples: int,
    tau_z: float,
    cfg_scale: float = 0.0,
    uncond_value: float = 0.0,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    decode_batch: int = 512,
    max_len: int = 102,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: torch.device | str = 'cuda',
    cfg_variant: str = 'standard',
    return_latents: bool = False,
    # --- normalization + conditioning ---
    z_mean: Optional[torch.Tensor] = None,
    z_std: Optional[torch.Tensor] = None,
    y_mean: Optional[torch.Tensor] = None,
    y_std: Optional[torch.Tensor] = None,):

    device = torch.device(device)
    latent_dim = 128
    generated_smiles: List[str] = []
    z0_all = []
    zt_all = []
    total_batches = (n_samples + decode_batch - 1) // decode_batch
    iterator = range(0, n_samples, decode_batch)
    iterator = tqdm(iterator, total = total_batches, desc = 'Sampling and decoding')
    tau_cond = float(tau_z)
    uncond_cond = float(uncond_value)
    if y_mean is not None and y_std is not None:
        tau_cond = (tau_cond - float(y_mean.item())) / float(y_std.item())
        uncond_cond = (uncond_cond - float(y_mean.item())) / float(y_std.item())

    for i in iterator:
        bs = min(decode_batch, n_samples - i)
        z0 = torch.randn(bs, latent_dim, device = device) # < --- Sample from prior
        c = torch.full((bs, 1), tau_cond, device=device)
        vfield = make_vfield(flow, c, guidance_scale=cfg_scale, uncond_value=uncond_cond, variant=cfg_variant)
        zt = rk45_integrator(vfield, z0, rtol = rtol, atol = atol)
        
        # if trained in standardized latent space, unstandardize before decoding
        zt_decode = zt
        if z_mean is not None and z_std is not None: zt_decode = zt * z_std + z_mean
        if return_latents:
            z0_all.append(z0.detach().cpu())
            zt_all.append(zt.detach().cpu())
        tok = vae.generate(latent_variable = zt_decode, max_len = max_len, temperature = temperature, top_p_val = top_p)
        for tokens in tok.detach().cpu().tolist():
            smi = tokenizer.decode_single(tokens); generated_smiles.append(smi)
    if return_latents: return generated_smiles, torch.cat(z0_all, dim=0), torch.cat(zt_all, dim=0)
    return generated_smiles
