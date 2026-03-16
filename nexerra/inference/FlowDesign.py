# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  Seeded inference on the flow model, additional hard-chemistry filters can be applied.
#   - read seed linker SMILES from ../../designed/linker/run/input.txt
#   - write filtered outputs to ../../designed/linker/run/output.txt
#   - write all generated outputs to ../../designed/linker/run/output_all.txt
#  Example usage:
#   python FlowDesign.py --alpha 0.9 --num-samples 1000 --batch-size 128 \
#   --reward gas --threshold 0.8 --filters True
#  Author: Dhruv Menon (dm958[at]cam[dot]ac[dot]uk)
#
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import argparse
import pickle
import math
import random; random.seed(42)
import numpy as np
if not hasattr(np, 'bool'): np.bool = np.bool_
from itertools import combinations
from tqdm import tqdm
from typing import List, Dict, Union, Tuple

# --- Logging ---
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PyTorch imports ---
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F

# --- Local imports ---
from nexerra.model.HTVAE import VAEModel
from nexerra.utils.tokenizer import Tokenizer
from nexerra.inference.Reward import RewardFunction
from nexerra.inference.Design import prepare_molecule, get_candidate_sites, placement_score, select_best_sites, place_lr_atoms, optimize_lr_placement
from nexerra.cfm.eval_utils import make_vfield, rk45_integrator

# --- SCScore ---
from nexerra.utils.scscore.scscore.standalone_model_numpy import SCScorer
logger.info("Synthetic complexity scorer loaded.")

# --- RDKit ---
from rdkit import RDLogger
# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem, AddHs
from rdkit.Chem.rdfiltercatalog import GetFunctionalGroupHierarchy
from rdkit.Chem.Crippen import MolLogP

import pyfiglet
def display_banner(): banner = pyfiglet.figlet_format("Nexerra", font="slant"); print(banner)

# --- Model setup: v_theta(t, z, c) ---
# -----------------------------------------------------------
# Exact same setup as otcfm_trainer.py.
# Do NOT change (you may get drastically different outcomes)
# -----------------------------------------------------------
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

class FiLM(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim * 2),
        )
    def forward(self, cond):
        gamma, beta = self.net(cond).chunk(2, dim = -1)
        return gamma, beta

class ResBlock(nn.Module):
    def __init__(self, hidden: int, cond_dim: int, film_hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.film = FiLM(dim = hidden, hidden = film_hidden)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, cond):
        h = self.ln(x)
        gamma, beta = self.film(cond)
        h = h * (1.0 + gamma) + beta
        h = self.fc1(F.silu(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h

class LatentCFM(nn.Module):
    '''
    Residual MLP with FiLM conditioning (matches training architecture)
    '''
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
        self.tembed = TimeFourier(64)
        cond_in = 128 + cond_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.in_proj = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList([
            ResBlock(hidden, cond_dim = hidden, film_hidden = film_hidden, dropout = dropout)
            for _ in range(n_blocks)
        ])
        self.out_ln = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, latent_dim)
        nn.init.normal_(self.out_proj.weight, mean = 0.0, std = 1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, t, z, c):
        tc = torch.cat([self.tembed(t), c], dim = -1)
        cond = self.cond_proj(tc)
        x = self.in_proj(z)
        for blk in self.blocks:
            x = blk(x, cond)
        x = self.out_ln(x)
        return self.out_proj(x)

# --- Generate (from seed) ---
class FlowDesign:
    def __init__(self,
                 device,
                 model: VAEModel,
                 flow: LatentCFM, 
                 tokenizer: Tokenizer,
                 config: Dict,
                 z_mean: torch.Tensor | None = None,
                 z_std: torch.Tensor | None = None,
                 standardize_latents: bool = True):
        
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.flow = flow.to(self.device)
        self.flow.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.standardize_latents = standardize_latents
        self.z_mean = z_mean.to(device) if isinstance(z_mean, torch.Tensor) else None
        self.z_std = z_std.to(device) if isinstance(z_std, torch.Tensor) else None

    def generate_from_seed(self, 
                           seed_smiles: str,
                           alpha: float,     
                           num_samples: int, 
                           batch_size: int) -> List[str]:
        return generate_smiles_from_seed(seed_smiles,
                                         alpha,
                                         flow = self.flow,
                                         vae = self.model,
                                         tokenizer = self.tokenizer,
                                         n_samples = int(num_samples),
                                         tau_z = float(self.config['tau_z']),
                                         cfg_scale = float(self.config.get('guidance_scale', 0.0)),
                                         uncond_value = float(self.config.get('uncond_value', 0.0)),
                                         solver = str(self.config.get('solver', 'rk45')).lower(),
                                         rtol = float(self.config.get('rtol', 1e-4)),
                                         atol = float(self.config.get('atol', 1e-6)),
                                         decode_batch = int(batch_size),
                                         max_len = int(self.config['max_len']),
                                         temperature = float(self.config.get('temperature', 0.8)),
                                         top_p = float(self.config.get('top_p', 0.9)),
                                         device = self.device,
                                         cfg_variant = str(self.config.get('cfg_variant', 'standard')),
                                         z_mean = self.z_mean if self.standardize_latents else None,
                                         z_std = self.z_std if self.standardize_latents else None)

@torch.no_grad()
def generate_smiles_from_seed(seed,
                              alpha,
                              flow,
                              vae,
                              tokenizer,
                              n_samples: int,
                              tau_z: float,
                              cfg_scale: float = 0.0,
                              uncond_value: float = 0.0,
                              solver: str = 'rk45',
                              rtol: float = 1e-4,
                              atol: float = 1e-6,
                              decode_batch: int = 512,
                              max_len: int = 102,
                              temperature: float = 0.8,
                              top_p: float = 0.9,
                              device: Union[torch.device, str] = 'cuda',
                              cfg_variant: str = 'standard',
                              return_latents: bool = False,
                              z_mean: torch.Tensor | None = None,
                              z_std: torch.Tensor | None = None,) -> Union[List[str], Tuple[List[str], torch.Tensor, torch.Tensor]]:
    
    device = torch.device(device)
    generated_smiles: List[str] = []
    z0_all = []
    zt_all = []
    
    # --- encode seed SMILES to latent space ---
    tokens = tokenizer.encode_single(seed).unsqueeze(0).to(device)
    src_key_padding_mask = (tokens == vae.pad_index).to(device)
    with torch.no_grad():
        _, mu, _ = vae.encode(
            src = tokens, src_mask = None, src_key_padding_mask = src_key_padding_mask)
    z_seed = mu.squeeze(0)
    if z_mean is not None and z_std is not None:
        z_mean = z_mean.to(device)
        z_std = z_std.to(device)
        z_seed = (z_seed - z_mean) / z_std
    # --- generate in batches ---
    for i in tqdm(range(0, n_samples, decode_batch), desc = "Generating Linkers"):
        bs = min(decode_batch, n_samples - i)
        # --- mix with prior noise per-sample to explore neighborhood ---
        z_seed_bs = z_seed.unsqueeze(0).repeat(bs, 1).to(device)  # [bs, latent_dim]
        eps_bs = torch.randn_like(z_seed_bs)
        z0_bs  = alpha * z_seed_bs + math.sqrt(max(1e-8, 1.0 - alpha ** 2)) * eps_bs
        c = torch.full((bs, 1), float(tau_z), device = device)
        vfield = make_vfield(flow, c, guidance_scale = cfg_scale, uncond_value = uncond_value, variant = cfg_variant)
        if solver != 'rk45': logger.warning("Solver '%s' not supported; using rk45", solver)
        zt = rk45_integrator(vfield, z0_bs, rtol = rtol, atol = atol)
        if return_latents:
            z0_all.append(z0_bs.detach().cpu())
            zt_all.append(zt.detach().cpu())
        zt_decode = zt
        if z_mean is not None and z_std is not None:
            zt_decode = zt * z_std + z_mean
        tok = vae.generate(latent_variable = zt_decode, max_len = max_len, temperature = temperature, top_p_val = top_p)
        for tokens in tok.detach().cpu().tolist():
            smi = tokenizer.decode_single(tokens); generated_smiles.append(smi)
    if return_latents: return generated_smiles, torch.cat(z0_all, dim = 0), torch.cat(zt_all, dim = 0)
    return generated_smiles

# ---- gating function(s) ----
'''Hard-chemistry filters -->
    1. Ring filters: remove molecules with 3- or 4-membered rings, or rings larger than 6
    2. Functional group filters: remove molecules with azides, diazo groups, nitroso groups, peroxides, sulfonates, phosphates, hydroxamic acids, carboxyl groups
    3. SCScore filter: remove molecules with SCScore above a threshold
Same as Design.py; a smart programmer would have separate modules for this, but alas, despite my best efforts, smartness evades me.
'''
def ring_filter(smiles: List[str], reject_three: bool = True, reject_four: bool = True, reject_large: bool = True) -> List[str]:
    '''also removes molecules with triple bonded carbons in rings (unable to form conformers)'''
    filtered = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi.replace("[Lr]", "*"))
        if mol is None: continue
        ri = mol.GetRingInfo(); atom_rings = ri.AtomRings()
        ring_lengths = [len(ring) for ring in atom_rings]
        if reject_three == True and 3 in ring_lengths: continue
        if reject_four == True and 4 in ring_lengths: continue
        if reject_large == True and any(r > 6 for r in ring_lengths): continue
        # --- Remove triple bonded carbons in rings ---
        ring_has_triple_bond = False
        for b in mol.GetBonds():
            if b.GetBondType() == Chem.BondType.TRIPLE and b.IsInRing(): ring_has_triple_bond = True; break
        if not ring_has_triple_bond: filtered.append(smi)
    return filtered

def fg_filter(smiles: List[str], azide: bool = True, diazo: bool = True, nitroso: bool = True, peroxide: bool = True,
              sulfonate: bool = True, phosphate: bool = True, hydroxamic: bool = True, carboxyl: bool = True) -> List[str]:
    '''Avoid unstable/reactive functional groups <-- destabilize(s) the linker OR
    may serve as additional coordination sites for metal centers
    Filters out azides, diazo groups, nitroso groups, and peroxides (by default), BUT
    can be toggled based on the needs'''
    
    _HAZARD_PREFIXES = []
    if azide == True: _HAZARD_PREFIXES.append("Azide")
    if diazo == True: _HAZARD_PREFIXES.append("Diazo")
    if nitroso == True: _HAZARD_PREFIXES.append("Nitroso")
    if peroxide == True: _HAZARD_PREFIXES.append("Peroxide")
    if sulfonate: _HAZARD_PREFIXES.append("Sulfonate")
    if phosphate: _HAZARD_PREFIXES.append("Phosphate")
    if hydroxamic: _HAZARD_PREFIXES.extend(["Hydroxamic", "Hydroxylamino"])
    if carboxyl: _HAZARD_PREFIXES.extend(["Carboxyl", "Carboxylic"])
    # --- set up filter catalog ---
    try:
        _FGCAT = GetFunctionalGroupHierarchy(); use_catalog = True
    except Exception:
        use_catalog = False; print("Warning: GetFunctionalGroupHierarchy not available, using SMARTS patterns")
    # --- SMARTS patterns for additional functional groups ---
    additional_smarts = {}
    if sulfonate: additional_smarts['sulfonate'] = ["S(=O)(=O)O"]
    if phosphate: additional_smarts['phosphate'] = ["P(=O)(O)(O)O"]
    if hydroxamic: additional_smarts['hydroxamic'] = ['[CX3](=O)N[OX2H1]']
    if carboxyl: additional_smarts['carboxylic'] = ['C(=O)O', 'C(=O)[OH]', '[CX3](=O)[OH]']
    # --- filter based hazardous functional groups ---
    filtered_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi.replace("[Lr]", "*"))
        if mol is None: continue
        has_hazard = False
        # --- check for hazardous functional groups matches using the hierarchy first ---
        if use_catalog:
            try:
                matches = _FGCAT.GetMatches(mol)
                _FGLABELS = [m.GetDescription() for m in matches]
                for label in _FGLABELS:
                    if any(label.startswith(prefix) for prefix in _HAZARD_PREFIXES): has_hazard = True; break
            except Exception: pass
        # --- switch to SMARTS pattern ---
        if not has_hazard:
            for _, patterns in additional_smarts.items():
                for pattern in patterns:
                    try:
                        pattern_mol = Chem.MolFromSmarts(pattern)
                        if pattern_mol is not None and mol.HasSubstructMatch(pattern_mol): has_hazard = True; break
                    except Exception: continue
                if has_hazard: break
        # --- if no hazardous groups, keep the molecule ---
        if not has_hazard: filtered_smiles.append(smi)
    return filtered_smiles

def scscore_filter(smiles: List[str], threshold: float = 4.5) -> List[str]:
    '''reject molecules with scscore > threshold'''
    scmodel = SCScorer()
    scmodel.restore(os.path.join('../utils','scscore', 'models', 'full_reaxys_model_2048bool', 'model.ckpt-10654.as_numpy.json.gz'), FP_rad = 2, FP_len = 2048)
    filtered = []
    for smi in smiles:
        clean_smi = smi.replace("[Lr]", "*")
        mol = Chem.MolFromSmiles(clean_smi)
        if mol is None: continue
        _, sco = scmodel.get_score_from_smi(clean_smi)
        if sco <= threshold: filtered.append(smi)
    return filtered

def reward_filter(rf: RewardFunction, toggle: int, smiles: List[str], threshold: float = 0.5) -> List[str]:
    '''Filter generated SMILES based on the reward function.'''
    # --- Filter based on reward ---
    filtered = []
    for smi in smiles:
        if toggle == 1:
            r = rf.R_grav(smi)
            if r >= threshold: filtered.append((smi, r))
        elif toggle == 0:
            r = rf.R_gas(smi)
            if r >= threshold: filtered.append((smi, r))
    filtered.sort(key = lambda x: x[1], reverse = True)  # Sort by reward
    filtered = [smi for smi, _ in filtered]
    return filtered

# -----------------------------------------------------------
# DO NOT CHANGE THIS BLOCK (unless you know what you're doing)
# To avoid too many moving parts, all relative paths are fixed
# All user-defined parameters go to the inference_config.txt file
# -----------------------------------------------------------    

def load_config(config_path: str = '../../designed/linker/inference_config.txt'):
    config = {}
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if value.lower() in ['true', 'false']: config[key] = value.lower()
                    elif value == 'auto': config[key] = 'auto'
                    elif '.' in value:
                        try: config[key] = float(value)
                        except ValueError: config[key] = value
                    else:
                        try: config[key] = int(value)
                        except ValueError: config[key] = value
    else:
        logger.warning("Config file not found at %s; using defaults and dataset-derived values.", config_path)
    return config

def auto_configs(config, dataset):
    # --- Provide dataset-derived defaults if missing ---
    if 'latent_dim' not in config or config['latent_dim'] == 'auto':
        config['latent_dim'] = 128 if 'latent_dim' not in dataset else dataset['latent_dim']
    if 'max_len' not in config or config['max_len'] == 'auto':
        config['max_len'] = dataset['max_len']
    return config

if __name__ == "__main__":
    display_banner()
    parser = argparse.ArgumentParser(description = 'Flow Inference')
    parser.add_argument('--alpha', type = float, default = 0.9, help = 'Alpha value for mixing seed latent with noise (default: 0.9)')
    parser.add_argument('--num-samples', type = int, default = 100, help = 'Number of samples to generate (default: 100)')
    parser.add_argument('--batch-size', type = int, default = 64, help = 'Batch size for decoding (default: 64)')
    parser.add_argument('--reward', type = str, default = 'gas', choices = ['gas', 'grav'], help = 'Reward function to use for design (default: gas)')

    # -------------------------- 
    # Configurable paths
    # DONT NEED TO CHANGE UNLESS YOU CHANGE RELATIVE PATHS
    # -------------------------- 
    
    parser.add_argument('--config', type = str, default = '../../designed/linker/inference_config.txt', help = 'Path to inference_config.txt')
    parser.add_argument('--vae-ckpt', type = str, default = '../../artifacts/ckpt/vae/no_prop_vae_epoch_120.pt', help = 'Path to VAE checkpoint')
    parser.add_argument('--flow-ckpt', type = str, default = '../../artifacts/ckpt/flow/otcfm_step_180000.pt', help = 'Path to flow checkpoint')
    parser.add_argument('--latent-bank', type = str, default = '../../artifacts/latent_banks/latent_bank_len.pt', help = 'Path to latent bank file')
    parser.add_argument('--threshold', type = float, default = 0.5, help = 'Property filter threshold (default: 0.5)')
    parser.add_argument('--filters', action = 'store_true', help = 'Apply chemistry filters to generated molecules')
    args = parser.parse_args()

    # -----------------------------------------------------------
    # DO NOT CHANGE THIS BLOCK (unless you know what you're doing)
    # To avoid too many moving parts, all relative paths are fixed
    # All user-defined parameters go to the inference_config.txt file
    # -----------------------------------------------------------    
    
    # --- Setup the VAE and Tokenizer first ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.vae_ckpt
    mparams = '../../data/processed/tokenized_dataset.pkl'
    with open(mparams, 'rb') as f: dataset = pickle.load(f)
    tok2id = dataset['tok2id']
    id2tok = dataset['id2tok']
    max_len = dataset['max_len']
    vocab_size = dataset['vocab_size']
    sos_index = dataset['sos_index']
    eos_index = dataset['eos_index']
    padding_index = dataset['padding_index']
    unk_index = dataset['unk_index']
    # ---- Inference Config ----
    inference_config = load_config(args.config)
    inference_config = auto_configs(inference_config, dataset)
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
        latent_dim = inference_config['latent_dim'],
        num_head = 8,
        num_encoder_layers = 6, 
        num_decoder_layers = 6,
        d_feedforward = 2048,
        encoder_dropout = 0.1, 
        decoder_dropout = 0.05, 
        max_len = max_len,
        activation = 'relu'
    )
    model.to(device)
    logger.info("Model initialized.")
    tokenizer = Tokenizer(tok2idx = tok2id, idx2tok = id2tok, max_len = max_len)
    logger.info("Tokenizer initialized.")
    try: checkpoint = torch.load(ckpt_path, map_location = device, weights_only = True)
    except TypeError: checkpoint = torch.load(ckpt_path, map_location = device)
    model.load_state_dict(checkpoint['model_state'])
    logger.info("Model state loaded from %s", ckpt_path)

    # --- Flow model initialization ---
    lb_path = args.latent_bank
    latent_bank = torch.load(lb_path, map_location = device)
    Y = latent_bank['Y'].cpu().numpy()
    percentile = 60.0
    tau_z = float(np.percentile(Y, percentile))
    inference_config['tau_z'] = tau_z
    logger.info("Latent bank loaded from %s", lb_path)
    logger.info("Setting tau_z (%.1f percentile) = %.4f", percentile, tau_z)

    flow_ckpt_path = args.flow_ckpt
    flow_ckpt = torch.load(flow_ckpt_path, map_location = device)

    if 'cutoff' in flow_ckpt:
        inference_config['tau_z'] = float(flow_ckpt['cutoff'])
        try:
            percentile = float(100.0 * (Y <= inference_config['tau_z']).mean())
            logger.info("Overriding tau_z from checkpoint: %.6f (approx percentile ~ %.2f%%)", inference_config['tau_z'], percentile)
        except Exception:
            logger.info("Overriding tau_z from checkpoint: %.6f", inference_config['tau_z'])

    if 'guidance_scale' in flow_ckpt: inference_config['guidance_scale'] = float(flow_ckpt['guidance_scale'])
    if 'uncond_value' in flow_ckpt: inference_config['uncond_value'] = float(flow_ckpt['uncond_value'])
    if 'solver' in flow_ckpt: inference_config['solver'] = str(flow_ckpt['solver']).lower()
    if 'rtol' in flow_ckpt: inference_config['rtol'] = float(flow_ckpt['rtol'])
    if 'atol' in flow_ckpt: inference_config['atol'] = float(flow_ckpt['atol'])
    if 'cfg_variant' in flow_ckpt: inference_config['cfg_variant'] = str(flow_ckpt['cfg_variant'])

    standardize_latents = bool(flow_ckpt.get('standardize_latents', True))
    z_mean = flow_ckpt.get('z_mean')
    z_std = flow_ckpt.get('z_std')
    if isinstance(z_mean, torch.Tensor): z_mean = z_mean.to(device)
    if isinstance(z_std, torch.Tensor): z_std = z_std.to(device)

    flow = LatentCFM(latent_dim = inference_config['latent_dim'], 
                     cond_dim = 1, hidden = 512).to(device)
    flow.load_state_dict(flow_ckpt["state_dict"]); flow.eval()
    logger.info("Flow model loaded from %s", flow_ckpt_path)
    
    # --- Setup the Designer class ---
    designer = FlowDesign(device = device, 
                          model = model, 
                          flow = flow, 
                          tokenizer = tokenizer, 
                          config = inference_config,
                          z_mean = z_mean,
                          z_std = z_std,
                          standardize_latents = standardize_latents)
    
    run_dir = '../../designed/linker/run/'
    input_path = run_dir + 'input.txt'
    output_path = run_dir + 'output.txt'
    output_path_all = output_path.replace('.txt', '_all.txt')
    assert os.path.isfile(input_path), f"Input file {input_path} does not exist. Please provide a valid input file."
    with open(input_path, 'r') as f:
        seed_smiles = f.read().strip()
    print(f"Input SMILES: {seed_smiles}")
    os.makedirs(run_dir, exist_ok = True)
    # --- Generate new molecules ---
    generated = designer.generate_from_seed(seed_smiles = seed_smiles,
                                            alpha = args.alpha,     
                                            num_samples = args.num_samples, 
                                            batch_size = args.batch_size)
    # --- Optimize [Lr] placement ---
    optimized_smiles = []
    num_connections = seed_smiles.count('[Lr]')
    for smi in generated:
        try:
            opt_smi = optimize_lr_placement(smi, num_connections = num_connections)
            optimized_smiles.append(opt_smi)
        except Exception as e:
            logger.warning("Failed to optimize [Lr] placement for %s: %s", smi, str(e))
    
    optimized_smiles = list(set(optimized_smiles)) 
    with open(output_path_all, 'w') as f: 
        for smi in optimized_smiles: f.write(smi + '\n')
    logger.info("All generated SMILES saved to %s", output_path_all)
    if args.filters:
        # --- Ring filter ---
        ring_optimized_smiles = ring_filter(optimized_smiles, reject_three = True, reject_four = True)
        logger.info("%d molecules remain after ring filters.", len(ring_optimized_smiles))
        # --- SCScore filter ---
        scs_ring_optimized_smiles =  scscore_filter(ring_optimized_smiles, threshold = 4.5)
        logger.info("%d molecules remain after SCScore filter.", len(scs_ring_optimized_smiles))
        # --- Functional group filter ---
        fg_scs_ring_optimized_smiles = fg_filter(scs_ring_optimized_smiles, azide = True, diazo = True, nitroso = True, peroxide = True,
                                                 sulfonate = True, phosphate = True, hydroxamic = True, carboxyl = True)
        logger.info("%d molecules remain after functional group filter.", len(fg_scs_ring_optimized_smiles))
        optimized_smiles = fg_scs_ring_optimized_smiles
    optimized_smiles = list(set(optimized_smiles))
    logger.info("%d UNIQUE molecules generated after [Lr] placement optimization & hard-chemistry filtering.", len(optimized_smiles))
    logger.info(f"Moving to reward filtering.")
    # --- Reward filter ---
    rf = RewardFunction()
    if args.reward == 'gas': toggle = 0
    elif args.reward == 'grav': toggle = 1
    else: raise ValueError(f"Choose a correct reward")
    filtered_smiles = reward_filter(rf, toggle = toggle, smiles = optimized_smiles, threshold = args.threshold if args.threshold else 0.5)
    filtered_smiles = list(set(filtered_smiles))
    logger.info("%d UNIQUE molecules remain after reward filter.", len(filtered_smiles))
    # --- Save to output file ---
    with open(output_path, 'w') as f:
        for smi in filtered_smiles: f.write(smi + '\n')
    logger.info("Generated SMILES saved to %s", output_path)
    logger.info(f"Stored in decreasing order of reward score.")
# -----------------------------------------------------------------
