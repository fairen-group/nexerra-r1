# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  Setup the trainer module for the OT-CFM model. 
#  Trains an OT-CFM in the latent space of the molecular transformer VAE.
#  Author: Dhruv Menon (dm958[at]cam[dot]ac[dot]uk)
#
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import math
import pickle
import numpy as np 
from tqdm import tqdm
import random
random.seed(42)
from typing import Optional, Tuple
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(handler)

import torch                    
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher as OT_CFM
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP as logP

from nexerra.model.HTVAE import VAEModel
from nexerra.utils.tokenizer import Tokenizer
from nexerra.inference.Reward import RewardFunction
from nexerra.cfm.eval_utils import generate_smiles as _gen
    

import pyfiglet
def display_banner(): banner = pyfiglet.figlet_format("Nexerra", font = "slant"); print(banner)

# -------------------------------------
# Fixed model setup 
# -------------------------------------
# --- Model setup: v_theta(t, z, c) ---
class TimeFourier(nn.Module):
    '''Fourier feature map for t in [0,1]
    Positional encoding'''
    def __init__(self, dim = 64):
        super().__init__()
        self.dim = dim
        # --- B: batch size ---
        self.register_buffer("B", torch.randn(1, self.dim), persistent = False)
    def forward(self, t):  # --- t:[B,1] in [0,1] ---
        if t.dim() == 1: t = t.unsqueeze(-1) # -> [B,1]
        t = t.to(self.B.dtype)
        # --- broadcasting: [B,1] * [1,dim] -> [B,dim] ---
        ang = 2 * math.pi * t * self.B
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim = -1)  # [B, 2 * dim]

# ------------- new module -------------
class FiLM(nn.Module):
    '''per-feature scale/shift from a conditioning vector
    ref: 10.1609/aaai.v32i1.11671'''
    def __init__(self, cond_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * out_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, cond):
        gamma, beta = self.net(cond).chunk(2, dim = -1)
        return gamma, beta

class ResBlock(nn.Module):
    '''arXiv:1512.03385v1'''
    def __init__(self, dim: int, cond_dim: int, film_hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.film = FiLM(cond_dim, film_hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        h = self.ln(x)
        gamma, beta = self.film(cond)
        # FiLM modulation (1 + gamma)
        h = h * (1.0 + gamma) + beta
        h = F.silu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h  # <-- residual

class LatentCFM(nn.Module):
    '''
    - Fourier time embedding
    - Separate embedding for (t, c)
    - Residual blocks with FiLM at each block
    ref:
    - arXiv:2210.02747v2; arXiv:2302.00482v4'''
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int = 1,
        hidden: int = 512,
        n_blocks: int = 8,
        film_hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.tembed = TimeFourier(64)  # should output 128 dims

        # build a conditioning vector used everywhere ---> [t_embed, c]
        cond_in = 128 + cond_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),)
        self.in_proj = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList([
            ResBlock(hidden, cond_dim = hidden, film_hidden = film_hidden, dropout = dropout)
            for _ in range(n_blocks)])
        self.out_ln = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, latent_dim)

        nn.init.normal_(self.out_proj.weight, mean = 0.0, std = 1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, t, z, c):
        # t: [B,1], z: [B,D], c: [B,cond_dim]
        tc = torch.cat([self.tembed(t), c], dim = -1)      # [B, 128 + cond_dim]
        cond = self.cond_proj(tc)                        # [B, hidden]
        x = self.in_proj(z)                              # [B, hidden]
        for blk in self.blocks:
            x = blk(x, cond)                             # FiLM each block
        x = self.out_ln(x)
        v = self.out_proj(x)                             # [B, D]
        return v

# --- Utils ---
def pick_matcher(name: str):
    '''The current compatible matcher is OT-CFM. 
    Alternatively, one could implement SB-CFM (but the performance was found to be sub-optimal)'''
    name = name.lower()
    assert name == "otcfm", f"Unknown matcher: {name}"
    return OT_CFM()
    
class PercentileScheduler:
    '''Linear scheduler for the percentile from start to end over total_steps
    ---
    For increasing the molecule length:
        -> The plan is to step from the 70th percentile to the 90th percentile over the course of training.
    For decreasing the logP:
        -> The plan is to step from the 30th percentile to the 10th percentile over the course of the training.
    '''
    def __init__(self, start: float, end: float, total_steps: int):
        self.start = start
        self.end = end
        self.total_steps = total_steps

    def p_at_step(self, step: int) -> float:
        if step >= self.total_steps: return self.end
        return self.start + (self.end - self.start) * (step / self.total_steps)


class TargetSampler:
    '''Sample (Z, Y) from a latent bank with optional thresholding
    Notes
    -----
    - mode = 'max': higher Y is better (e.g. length); we sort descending
    - mode = 'min': lower Y is better (e.g. logP if minimizing); we sort ascending

    The sampler supports:
      - sample(): sample uniformly from the top-percentile subset
      - sample_min_value(): additionally enforce a hard Y-threshold
    '''

    def __init__(self, Z: torch.Tensor, Y: torch.Tensor, mode: str = 'max'):
        self.Z = Z
        self.Y = Y.squeeze()
        self.N = int(self.Y.numel())
        self.mode = mode

        # sort indices by Y
        if self.mode == 'min': self.order = torch.argsort(self.Y, descending = False)
        else: self.order = torch.argsort(self.Y, descending = True)

        self.Ys = self.Y[self.order]
        self.Zs = self.Z[self.order]
        self._rng = np.random.default_rng(seed = 42)

    def _topk(self, percentile: float) -> int:
        top_frac = (100.0 - float(percentile)) / 100.0
        k = int(self.N * top_frac)
        k = max(2, min(k, self.N))
        return k

    def sample(self, batch: int, percentile: float, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Uniformly sample from the percentile subset. Returns (z, y)'''
        k = self._topk(percentile)
        if k < batch: # if num samples < batch --> sample with replacement
            rel_idx = self._rng.choice(np.arange(k), size = batch, replace = True)
        else:
            rel_idx = self._rng.choice(np.arange(k), size = batch, replace = False)
        rel_idx = torch.from_numpy(rel_idx).long()
        z = self.Zs[rel_idx]
        y = self.Ys[rel_idx].unsqueeze(-1)
        return z.to(device), y.to(device)

    def sample_min_value(self, batch: int, percentile: float, device: str, min_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Sample from the top-percentile subset subject to a hard Y threshold.
        For mode = 'max': enforces y >= min_value
        For mode = 'min': enforces y <= min_value
        If the constraint is too strict (no eligible points in the pool),
        we fall back to sampling from the percentile pool (still stable, but weaker)'''
        
        k = self._topk(percentile)
        Ys_top = self.Ys[:k]
        if self.mode == 'min':
            # Ys_top is ascending; eligible are prefix where y <= min_value
            eligible = torch.nonzero(Ys_top <= float(min_value), as_tuple = False).view(-1)
        else:
            # Ys_top is descending; eligible are prefix where y >= min_value
            eligible = torch.nonzero(Ys_top >= float(min_value), as_tuple = False).view(-1)

        if eligible.numel() < 2:
            # fallback: no/too few eligible points -> sample from percentile pool
            return self.sample(batch = batch, percentile = percentile, device = device)

        e = int(eligible.numel())
        if e < batch: rel_idx = self._rng.choice(np.arange(e), size = batch, replace = True)
        else: rel_idx = self._rng.choice(np.arange(e), size = batch, replace = False)

        rel_idx = torch.from_numpy(rel_idx).long()
        z = self.Zs[rel_idx]
        y = self.Ys[rel_idx].unsqueeze(-1)
        return z.to(device), y.to(device)

# --- Evaluation -> sample from the trained flow to evaluate the performance of the model ---
@torch.no_grad()
def metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    '''Compute rRMSE and cosine similarity between pred and target'''
    mse = torch.mean((pred - target) ** 2)
    denom = torch.mean(torch.sum(target ** 2, dim = -1)) + eps
    rrmse = torch.sqrt(mse / denom)

    dot = torch.sum(pred * target, dim = -1)
    n1  = torch.norm(pred, dim = -1) + eps
    n2  = torch.norm(target, dim = -1) + eps
    cosine = torch.mean(dot / (n1 * n2))
    return rrmse.detach().item(), cosine.detach().item()

@torch.no_grad()
def eval(
    step:           int,
    flow:           nn.Module,
    vae:            VAEModel,
    tokenizer:      Tokenizer,
    cutoff:         Optional[float],
    log_dir:        str,
    device:         str,
    n_samples:      int = 2000,
    decode_batch:   int = 512,
    rtol:           float = 1e-4,
    atol:           float = 1e-6,
    cfg_scale:      float = 0.0,
    uncond_value:   float = 0.0,
    tag:            Optional[str] = None,
    save_smiles:    bool = True,
    max_len:        int = 102, 
    temperature:    float = 0.8, 
    top_p:          float = 0.9,
    mode:           str = 'max',
    # normalization + conditioning
    z_mean:         Optional[torch.Tensor] = None,
    z_std:          Optional[torch.Tensor] = None,
    y_mean:         Optional[torch.Tensor] = None,
    y_std:          Optional[torch.Tensor] = None,):
    
    '''Module to evaluate the performance of the flow model during training & at the end of training
    ---
    Args: - step: current training step
          - flow: the trained flow model
          - vae: the pretrained VAE model
          - tokenizer: the tokenizer used for the VAE+
          - cutoff: the cutoff value used for binary conditioning
          - log_dir: directory to save the evaluation results
          - device: device 
          - n_samples: number of samples to generate for evaluation
          - decode_batch: batch size for decoding the latents to SMILES
          - rk4_steps: number of RK4 steps for sampling
          - max_len, temperature, top_p: decoding parameters for the VAE
    ---
    Returns: None, saves the evaluation results to log_dir
    '''
    assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist"
    assert cutoff is not None, "Cutoff must be provided for evaluation"

    # --- Generate SMILES under the exact same inference path used elsewhere ---
    decoded_smiles, z0_tensor, zt_tensor = _gen(
        flow = flow,
        vae = vae,
        tokenizer = tokenizer,
        n_samples = int(n_samples),
        tau_z = float(cutoff),
        cfg_scale = float(cfg_scale or 0.0),
        uncond_value = float(uncond_value or 0.0),
        rtol = float(rtol),
        atol = float(atol),
        decode_batch = int(decode_batch),
        max_len = int(max_len),
        temperature = float(temperature),
        top_p = float(top_p),
        device = device,
        cfg_variant = 'standard',
        return_latents = True,
        z_mean = z_mean,
        z_std = z_std,
        y_mean = y_mean,
        y_std = y_std,
    )
    logger.info(f"Decoded {len(decoded_smiles)} / {n_samples} samples")
    
    # --- dump decoded SMILES for downstream eval scripts ---
    tag_sfx = f"_{tag}" if tag else ""
    if save_smiles:
        smiles_path = os.path.join(log_dir, f"flow_smiles_step_{step}{tag_sfx}.txt")
        with open(smiles_path, 'w') as f: 
            for smi in decoded_smiles: f.write((smi or '') + "\n")
        logger.info(f"Wrote decoded SMILES to {smiles_path}")

    # --- Compute rewards for the decoded SMILES ---
    '''Note: This part will need to be edited based on the reward function. 
    In future iterations this can be made more modularised.
    '''
    rf = RewardFunction()
    rewards = []
    valid_smiles = []
    for smi in decoded_smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None: 
                r1 = rf.anc_to_anc_length(smi = smi, try_3d = True); rewards.append(r1); valid_smiles.append(smi)
                # r1 = logP(mol); rewards.append(r1); valid_smiles.append(smi)
            else: rewards.append(None)
        except Exception as e:
            logger.error(f"Error decoding SMILES {smi}: {e}")
            rewards.append(None)

    assert len(valid_smiles) != 0, "No valid SMILES decoded --> check if everything is working fine"
    valid_rewards = [r1 for r1 in rewards if r1 is not None and not np.isnan(r1)]
    valid_rewards = np.array(valid_rewards, dtype = np.float64)
    # --- bin the rewards values --- 
    counts, edges = np.histogram(valid_rewards, bins = 30)
    # --- Compute rmse & cosine similarity between zt and z0 ---
    rmse, cosine = metrics(zt_tensor, z0_tensor, eps = 1e-8)
    # --- Save the results ---
    with open(os.path.join(log_dir, f"hist_step_{step}{tag_sfx}.txt"), "w") as f:
        f.write(f"# step: {step}\n")
        f.write(f"# n_valid: {len(valid_rewards)}\n")
        f.write(f"# cutoff: {float(cutoff):.6f}\n")
        f.write(f"# cfg_scale: {float(cfg_scale):.6f}\n")
        if mode == 'min': success = float((valid_rewards <= cutoff).mean())
        else: success = float((valid_rewards >= cutoff).mean())
        f.write(f"# success_at_tau: {success:.6f}\n")
        f.write(f"# median_reward: {float(np.median(valid_rewards)):.6f}\n")
        f.write(f"# p25_reward: {float(np.percentile(valid_rewards, 25)):.6f}\n")
        f.write(f"# p75_reward: {float(np.percentile(valid_rewards, 75)):.6f}\n")
        f.write(f"# rmse: {rmse:.6f}\n")
        f.write(f"# cosine: {cosine:.6f}\n")
        f.write("# bin_left,bin_right,count\n")
        for i, c in enumerate(counts): f.write(f"{edges[i]:.6f},{edges[i+1]:.6f},{int(c)}\n")
    logger.info(f"Wrote histogram for step {step}{tag_sfx} to {log_dir}")
    
# --- An evaluation sweep across multiple guidance scales ---
@torch.no_grad()
def eval_sweep(
    ckpt_path: str,
    vae: VAEModel,
    tokenizer: Tokenizer,
    device: str,
    log_dir: Optional[str] = None,
    cfg_scales = (0.0, 1.0, 1.5, 2.0, 3.0),
    n_samples: int = 10000,
    decode_batch: int = 512,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    uncond_value: float = 0.0,
    max_len: int = 102,
    temperature: float = 0.8,
    top_p: float = 0.9,
    mode: str = 'max',
):
    '''Run eval across multiple guidance scales for a saved flow checkpoint'''
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location = device)
    latent_dim = int(ckpt["latent_dim"])
    cutoff = float(ckpt.get("cutoff", 0.0))
    flow = LatentCFM(latent_dim = latent_dim, cond_dim = 1, hidden = 512, n_blocks = 8, film_hidden = 256, dropout = 0.0).to(device)
    try: state = ckpt["ema_state_dict"]
    except:
        print("EMA state dict not found, loading regular state dict")
        state = ckpt["state_dict"]
    flow.load_state_dict(state); flow.eval()

    step = int(ckpt.get("steps", 0))
    if log_dir is None: log_dir = os.path.dirname(ckpt_path) or "."
    os.makedirs(log_dir, exist_ok=True)

    # --- normalization + conditioning metadata ---
    standardize_latents = bool(ckpt.get("standardize_latents", True))
    z_mean = ckpt.get("z_mean")
    z_std = ckpt.get("z_std")
    y_mean = ckpt.get("y_mean")
    y_std = ckpt.get("y_std")
    if isinstance(z_mean, torch.Tensor): z_mean = z_mean.to(device)
    if isinstance(z_std, torch.Tensor): z_std = z_std.to(device)
    if isinstance(y_mean, torch.Tensor): y_mean = y_mean.to(device)
    if isinstance(y_std, torch.Tensor): y_std = y_std.to(device)
    
    logger.info(f"Eval sweep at step = {step}, cutoff = {cutoff:.4f}")
    for s in cfg_scales:
        tag = f"cfg_{s:.2f}"
        logger.info(f"Evaluating cfg_scale={s:.2f}")
        
        eval(
            step = step,
            flow = flow,
            vae = vae,
            tokenizer = tokenizer,
            cutoff = cutoff,
            log_dir = log_dir,
            device = device,
            n_samples = n_samples,
            decode_batch = decode_batch,
            rtol = rtol,
            atol = atol,
            cfg_scale = float(s),
            uncond_value = uncond_value,
            tag = tag,
            save_smiles = True,
            max_len = max_len,
            temperature = temperature,
            top_p = top_p,
            mode = mode,
            z_mean = (z_mean if standardize_latents else None),
            z_std = (z_std if standardize_latents else None),
            y_mean = y_mean,
            y_std = y_std,
        )

# -------------------------------------
# Core trainer loop; has been adequately described in the paper 
# -------------------------------------
def main(latent_pt: str,
         config: dict,
         model: VAEModel, 
         tokenizer: Tokenizer,
         mode: str = 'max',
         device: str = "cuda", 
         out_path: str = "otcfm_latent.pt"):
    
    # --- Unpack config ----
    batch = config['batch']
    matcher_name = config['matcher_name']
    steps = config['steps']
    lr = config['lr']
    grad_clip = config['grad_clip']
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
    # --- CFG + solver params ---
    cond_drop_prob = float(config.get('cond_drop_prob', 0.1))
    guidance_scale = float(config.get('guidance_scale', 2.0))
    rtol = float(config.get('rtol', 1e-4))
    atol = float(config.get('atol', 1e-6))
    uncond_value = float(config.get('uncond_value', 0.0))

    # ---- target hard-shift in Y (e.g., length) ----
    # For mode='max' (increase property): enforce y >= tau + delta_y
    # For mode='min' (decrease property): enforce y <= tau - delta_y
    
    delta_y = float(config.get('delta_y', 0.0))
    # ---- Setup ckpt directory ----
    ckpt_dir = os.path.dirname(out_path) or "."
    os.makedirs(ckpt_dir, exist_ok = True)
    
    # ---- Load latent bank ---- 
    data = torch.load(latent_pt, map_location = "cpu")
    Z, Y = data["Z"].float(), data["Y"].float()   # Z:[N,D], Y:[N]
    D = Z.shape[1]; assert D == 128, "Latent dim must be 128"
    logger.info(f"Loaded latent bank from {latent_pt} with {Z.shape[0]} samples")

    # ---- Latent/reward normalization and conditioning ----
    standardize_latents: bool = bool(config.get('standardize_latents', True)) 
    
    # ---- Compute stats ----
    z_mean = Z.mean(dim = 0)
    z_std = Z.std(dim = 0).clamp_min(1e-6)
    y_mean = Y.mean()
    y_std = Y.std().clamp_min(1e-6)

    if standardize_latents:
        Z = (Z - z_mean) / z_std
        logger.info("Standardized latents with dataset mean/std")

    # ---- Setup the tau scheduler & target sampler ----
    tau_scheduler = PercentileScheduler(start = config['percentile_start'], end = config['percentile_end'], total_steps = steps * 0.95)
    target_sampler = TargetSampler(Z, Y, mode)
    
    # unconditional sampler for CFG contrast (full bank via percentile = 0)
    uncond_percentile = float(config.get('uncond_percentile', 0.0))
    uncond_sampler = TargetSampler(Z, Y, mode)

    # --- empirical prior: sample z0 from the (optionally broad) latent bank ----
    # This keeps x0 on-manifold and makes OT transport meaningful for molecular design.
    prior_percentile = float(config.get('prior_percentile', uncond_percentile))
    def prior(n: int):
        z0, _ = uncond_sampler.sample(batch = n, percentile = prior_percentile, device = device)
        return z0

    # ---- setup model & Optimizer ---- 
    flow = LatentCFM(latent_dim = D, cond_dim = 1, hidden = 512, n_blocks = 8, film_hidden = 256, dropout = 0.0).to(device)
    optimizer  = torch.optim.AdamW(flow.parameters(), lr = lr)
    matcher = pick_matcher(matcher_name)

    # ---- LR scheduler: warmup + cosine ----
    warmup_steps = int(config.get('warmup_steps', max(1000, steps * 0.05)))
    # if cosine scheduler is needed --> min_lr = float(config.get('min_lr', 1e-5))
    def lr_lambda(s):
        if s < warmup_steps: return max(s, 1) / max(warmup_steps, 1)
        '''progress = (s - warmup_steps) / max(1, (steps - warmup_steps))
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        scale = (min_lr / lr) + (1 - (min_lr / lr)) * cos
        return scale'''
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- ema for stability (standard implementation) ----
    ema_decay = float(config.get('ema_decay', 0.999))
    ema_params = [p.detach().clone() for p in flow.parameters()]
    def ema_update(model):
        with torch.no_grad():
            for ep, p in zip(ema_params, model.parameters()):
                ep.mul_(ema_decay).add_(p.detach(), alpha=(1.0 - ema_decay))
    def load_ema_into(model):
        with torch.no_grad(): 
            for p, ep in zip(model.parameters(), ema_params): p.copy_(ep)

    vel_reg = float(config.get('velocity_reg', 0.0))
    flow.train()

    for step in tqdm(range(1, steps + 1), desc = "OT-CFM Training"):
        # --- Calculate current percentile & corresponding tau ---
        p = tau_scheduler.p_at_step(step)
        top_frac = (100.0 - p) / 100.0 
        k = int(top_frac * target_sampler.N)
        k = max(1, min(k, target_sampler.N - 1))
        tau = Y[target_sampler.order[k-1]].item() 
        # Conditional samples (top slice) and unconditional samples (broader slice) for CFG contrast
        min_target = (tau - delta_y) if (mode == 'min') else (tau + delta_y)
        z1_cond, y1_cond = target_sampler.sample_min_value(batch, p, device, min_value = min_target)
        z1_uncond, y1_uncond = uncond_sampler.sample(batch, uncond_percentile, device = device)

        # --- Sample z0 ~ N(0,I) ---
        z1_cond = z1_cond.to(device)
        z1_uncond = z1_uncond.to(device)
        y1_cond = y1_cond.to(device)
        y1_uncond = y1_uncond.to(device)
        n = z1_cond.size(0)
        z0 = prior(n)
        # Build per-branch conditions (normalize conditioning scalar for stable FiLM)
        tau_n = (float(tau) - float(y_mean.item())) / float(y_std.item())
        uncond_n = (float(uncond_value) - float(y_mean.item())) / float(y_std.item())
        c_cond = torch.full((n, 1), tau_n, device = device)
        c_uncond = torch.full_like(c_cond, uncond_n)
        # --- classifier-free guidance training via conditional dropout with data contrast ---
        '''If cond_drop_prob = 0.0 --> pure conditional (no CFG)'''
        if cond_drop_prob > 0.0: drop_mask = (torch.rand(n, 1, device=device) < cond_drop_prob)
        else: drop_mask = torch.zeros(n, 1, device = device, dtype = torch.bool)
        '''Anywhere the mask is applied, its unconditional & when not --> conditional'''
        mask_z = drop_mask.expand_as(z1_cond)
        mask_c = drop_mask.expand_as(c_cond)
        z1 = torch.where(mask_z, z1_uncond, z1_cond)
        c = torch.where(mask_c, c_uncond, c_cond)

        # --- OT-CFM minibatch coupling, returns (t, x_t, v_t) along OT displacement path ---
        t, x_t, v_t = matcher.sample_location_and_conditional_flow(x0 = z0, x1 = z1)
        pred = flow(t, x_t, c)
        loss = F.mse_loss(pred, v_t)
        if vel_reg > 0.0: loss = loss + vel_reg * pred.pow(2).mean()
        
        # --- Backpropagation ---
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        if grad_clip is not None: torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        ema_update(flow)

        if step % 20000 == 0: print(f"[{step}/{steps}] loss={loss.item():.5f}, percentile={p:.1f}%, tau={tau:.3f}")

        # --- Save checkpoint every 20000 steps ---
        if step % 20000 == 0:
            with torch.no_grad():
                # --- Evaluate with EMA weights ---
                emaflow = LatentCFM(latent_dim = D, cond_dim = 1, hidden = 512, n_blocks = 8, film_hidden = 256, dropout = 0.0).to(device)
                emaflow.load_state_dict(flow.state_dict())
                load_ema_into(emaflow)
                emaflow.eval()
                # --- Generate samples and evaluate ---
                eval(step = step,
                     flow = emaflow,
                     vae = model,
                     tokenizer = tokenizer,
                     cutoff = tau,
                     log_dir = ckpt_dir,
                     device = device,
                     n_samples = 5000,
                     decode_batch = 512,
                     rtol = rtol,
                     atol = atol,
                     cfg_scale = guidance_scale,
                     uncond_value = uncond_value,
                     mode = mode,
                     max_len = 102, temperature = 0.8, top_p = 0.8,
                     z_mean = (z_mean.to(device) if standardize_latents else None),
                     z_std = (z_std.to(device) if standardize_latents else None),
                     y_mean = y_mean.to(device),
                     y_std = y_std.to(device),
                     )

            ckpt_path = os.path.join(ckpt_dir, f"otcfm_step_{step}.pt")
            torch.save(
                {
                    "state_dict": flow.state_dict(),
                    "ema_state_dict": emaflow.state_dict(),
                    "latent_dim": D,
                    "cutoff": tau,
                    "matcher": matcher_name,
                    "steps": step,
                    "batch": batch,
                    "lr": lr,
                    "seed": seed,
                    "prior_type": "empirical",
                    "prior_percentile": prior_percentile,
                    "delta_y": delta_y,
                    "cond_drop_prob": cond_drop_prob,
                    "guidance_scale": guidance_scale,
                    "rtol": rtol,
                    "atol": atol,
                    "uncond_value": uncond_value,
                    # normalization metadata
                    "standardize_latents": standardize_latents,
                    "z_mean": z_mean.cpu(),
                    "z_std": z_std.cpu(),
                    "y_mean": y_mean.cpu(),
                    "y_std": y_std.cpu(),
                },
                ckpt_path)
            logger.info(f"Checkpoint saved at step {step}")

    # ---- Save final checkpoint ----
    emaflow = LatentCFM(latent_dim = D, cond_dim = 1, hidden = 512, n_blocks = 8, film_hidden = 256, dropout = 0.0).to(device)
    emaflow.load_state_dict(flow.state_dict())
    load_ema_into(emaflow)
    emaflow.eval()
    final_ckpt = out_path
    torch.save(
        {
            "state_dict": flow.state_dict(),
            "ema_state_dict": emaflow.state_dict(),
            "latent_dim": D,
            "cutoff": tau,
            "matcher": matcher_name,
            "device": str(device),
            "steps": steps,
            "batch": batch,
            "lr": lr,
            "seed": seed,
            "cond_drop_prob": cond_drop_prob,
            "guidance_scale": guidance_scale,
            "rtol": rtol,
            "atol": atol,
            "uncond_value": uncond_value,
            "prior_type": "empirical",
            "prior_percentile": prior_percentile,
            "delta_y": delta_y,
            # normalization metadata
            "standardize_latents": standardize_latents,
            "z_mean": z_mean.cpu(),
            "z_std": z_std.cpu(),
            "y_mean": y_mean.cpu(),
            "y_std": y_std.cpu(),
        },
        final_ckpt)
    logger.info("Training complete.")
    logger.info(f"Trained flow saved at {out_path}")

if __name__ == "__main__":
    display_banner()
    p = argparse.ArgumentParser()
    p.add_argument("--latent_pt", required = False, type=str)
    p.add_argument("--mode", type = str, choices = ['min', 'max'], default = 'max', help = "Whether to minimize or maximize the target property")
    p.add_argument("--percentile_start", type = float, required = False, default = None, help = 'Percentile of points to consider for binary conditional')
    p.add_argument("--percentile_end", type = float, required = False, default = None, help = 'Percentile of points to consider for binary conditional')
    p.add_argument("--steps", type = int, default = 50000)
    p.add_argument("--batch", type = int, default = 4096)
    p.add_argument("--matcher", type = str, default = "otcfm", choices = ["otcfm"])
    p.add_argument("--out_path", type = str, default = "otcfm_length.pt")
    p.add_argument("--lr", type = float, default = 1e-4)
    p.add_argument("--grad_clip", type = float, default = 1.0)
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--warmup_steps", type = int, default = None, help = "Number of warmup steps for LR scheduler")
    p.add_argument("--ema_decay", type = float, default = 0.999, help = "EMA decay for model parameters")
    p.add_argument("--velocity_reg", type = float, default = 0.0, help = "Velocity regularization weight")
    p.add_argument("--standardize_latents", action = argparse.BooleanOptionalAction, default = True, help = "Whether to standardize latents before training")
    # --- Training-time CFG + solver flags ---
    p.add_argument("--cond_drop_prob", type = float, default = 0.2, help = "Conditional dropout prob for CFG training")
    p.add_argument("--guidance_scale", type = float, default = 2.0, help = "CFG guidance scale used in periodic eval during training")
    p.add_argument("--uncond_percentile", type = float, default = 10.0, help = "Percentile for unconditional branch sampling (30.0 default)")
    p.add_argument("--rtol", type = float, default = 1e-4, help = "Relative tolerance for RK45")
    p.add_argument("--atol", type = float, default = 1e-6, help = "Absolute tolerance for RK45")
    p.add_argument("--uncond_value", type = float, default = 0.0, help = "Unconditional condition value for CFG (usually 0.0)")
    # --- Eval-only options ---
    p.add_argument("--eval_ckpt", type=str, default=None, help="Path to flow checkpoint (.pt) to evaluate")
    p.add_argument("--eval_scales", type=str, default="0.0,1.0,1.5,2.0,3.0", help="Comma-separated guidance scales for sweep")
    p.add_argument("--eval_samples", type=int, default=10000)
    p.add_argument("--eval_rtol", type=float, default=1e-4)
    p.add_argument("--eval_atol", type=float, default=1e-6)
    p.add_argument("--eval_uncond_value", type=float, default=0.0)
    p.add_argument("--eval_decode_batch", type=int, default=512)
    p.add_argument("--eval_log_dir", type=str, default=None)
    args = p.parse_args()

    # --- If eval-only, we don't require training arguments ---
    if args.eval_ckpt is None:
        assert args.latent_pt is not None, "--latent_pt is required for training"
        assert args.percentile_start is not None and args.percentile_end is not None, "--percentile must be set"
    # ---- Setup device -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ---- Load the pretrained VAE model & tokenizer ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.abspath(os.path.join(script_dir, '../../artifacts/ckpt/vae/no_prop_vae_epoch_120.pt'))
    mparams = os.path.abspath(os.path.join(script_dir, '../../data/processed/tokenized_dataset.pkl'))
    with open(mparams, 'rb') as f: dataset = pickle.load(f)
    tok2id = dataset['tok2id']
    id2tok = dataset['id2tok']
    max_len = dataset['max_len']
    vocab_size = dataset['vocab_size']
    sos_index = dataset['sos_index']
    eos_index = dataset['eos_index']
    padding_index = dataset['padding_index']
    unk_index = dataset['unk_index']
    
    # ---- Model initialization ----
    model = VAEModel(
        device = device,
        shape_flag = 0,
        vocab_size = vocab_size,
        sos_index = sos_index,
        eos_index = eos_index,
        pad_index = padding_index,
        unk_index = unk_index,
        d_model = 512,
        latent_dim = 128,
        num_head = 8,
        num_encoder_layers = 6, # 8
        num_decoder_layers = 6,
        d_feedforward = 2048,
        encoder_dropout = 0.1, # 0.1
        decoder_dropout = 0.05, # 0.0
        max_len = max_len,
        activation = 'relu'
    )
    model.to(device)
    logger.info("Model initialized.")
    tokenizer = Tokenizer(tok2idx = tok2id, idx2tok = id2tok, max_len=max_len)
    logger.info("Tokenizer initialized.")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only = True)
    model.load_state_dict(checkpoint['model_state'])
    logger.info("Model state loaded from %s", ckpt_path)

    # ---- Eval-only pathway ----
    if args.eval_ckpt is not None:
        scales = tuple(float(x.strip()) for x in args.eval_scales.split(",") if x.strip() != "")
        logger.info(f"Running eval sweep on {args.eval_ckpt} with scales={scales}")
        eval_sweep(
            ckpt_path = args.eval_ckpt,
            vae = model,
            tokenizer = tokenizer,
            device = device,
            log_dir = args.eval_log_dir,
            cfg_scales = scales,
            n_samples = int(args.eval_samples),
            decode_batch = int(args.eval_decode_batch),
            rtol = float(args.eval_rtol),
            atol = float(args.eval_atol),
            uncond_value = float(args.eval_uncond_value),
            max_len = max_len,
            temperature = 0.8,
            top_p = 0.8,
            mode = args.mode,
        )
    else:
        # ---- Configure the training parameters ----
        config = {
            'percentile_start' : float(args.percentile_start),
            'percentile_end' : float(args.percentile_end),
            'steps' : int(args.steps) if args.steps else 50000,
            'batch' : int(args.batch) if args.batch else 4096,
            'lr' : float(args.lr) if args.lr else 1e-4,
            'matcher_name' : args.matcher if args.matcher else "otcfm",
            'device' : device,
            'seed' : int(args.seed),
            'grad_clip' : float(args.grad_clip),
            'out_path' : "otcfm_latent.pt",
            'standardize_latents': bool(args.standardize_latents),
            'warmup_steps': args.warmup_steps if args.warmup_steps else max(1000, int(args.steps * 0.05)),
            'ema_decay': args.ema_decay if args.ema_decay else 0.999,
            'velocity_reg': args.velocity_reg if args.velocity_reg else 0.0,
            # CFG + solver
            'cond_drop_prob': float(args.cond_drop_prob),
            'guidance_scale': float(args.guidance_scale),
            'uncond_percentile': float(args.uncond_percentile),
            'rtol': float(args.rtol),
            'atol': float(args.atol),
            'uncond_value': float(args.uncond_value),
        }

        # ---- Start training ----
        logger.info(f"Starting the training")
        main(latent_pt = args.latent_pt if args.latent_pt else "latent_bank.pt",
             config = config,
             model = model,
             mode = args.mode,
             tokenizer = tokenizer,
             device = device,
             out_path = args.out_path if args.out_path else "otcfm_latent.pt")

    # --- Fin ---
