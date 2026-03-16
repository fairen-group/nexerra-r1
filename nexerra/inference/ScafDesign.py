# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  Scaffold-constrained design - in the input provid: 
#   - A core scaffold with multiple [Lr] anchors (first line)
#   - A ditopic arm with [Lr] anchors (second line)
#  Example usage: 
#  python ScafDesign.py --delta 0.2 --mode design --filters --reward gas --output ../../designed/linker/run/output.csv
#  python ScafDesign.py --mode test --reward grav --output ../../designed/linker/run/output.csv
#  Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

from __future__ import annotations

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Standard imports ---
from typing import List
import torch
from tqdm import tqdm
import argparse
import pickle
from itertools import combinations
import logging
import random
import pandas as pd
import numpy as np
if not hasattr(np, 'bool'): 
    np.bool = np.bool_
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- RDKit imports ---
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdfiltercatalog import GetFunctionalGroupHierarchy
from rdkit.Chem import rdmolops as RDMolOps

# --- Local imports ---
from nexerra.model.HTVAE import VAEModel
from nexerra.utils.tokenizer import Tokenizer
from nexerra.inference.Reward import RewardFunction
from nexerra.inference.Design import prepare_molecule, get_candidate_sites, placement_score, select_best_sites, place_lr_atoms, optimize_lr_placement

# --- SCScore ---
from nexerra.utils.scscore.scscore.standalone_model_numpy import SCScorer
logger.info("Synthetic complexity scorer loaded.")

import pyfiglet
def display_banner():
    banner = pyfiglet.figlet_format("Nexerra", font="slant")
    print(banner)

ANCHOR_SYMBOL = "Lr"  # anchor symbol in SMILES, used as a handle
# --- Grafting function ---
def graft_arm_to_core_symmetric(scaffold: Chem.Mol, arm: Chem.Mol) -> Chem.Mol:
    '''Symmetrically graft arms to all scaffold [Lr] positions
    ---
    Args: - scaffold (Chem.Mol): Scaffold molecule with [Lr] anchors
          - arm (Chem.Mol): Arm molecule with [Lr] anchors
    ---
    Returns: Chem.Mol: Resulting molecule with arms grafted to scaffold
    '''
    # --- Find scaffold [Lr] positions and neighbors --- 
    scaffold_connections = []
    for atom in scaffold.GetAtoms():
        if atom.GetSymbol() == ANCHOR_SYMBOL:
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neighbors) == 1: scaffold_connections.append((atom.GetIdx(), neighbors[0]))
    
    # --- Start with scaffold without [Lr] ---
    linker = Chem.RWMol(scaffold)
    # --- Remove scaffold [Lr] anchors (in reverse order) ---
    lr_indices = [idx for idx, _ in scaffold_connections]
    for lr_idx in sorted(lr_indices, reverse = True): linker.RemoveAtom(lr_idx)
    # --- Adjust connection indices after removals ---
    adjusted_connections = []
    for lr_idx, neighbor_idx in scaffold_connections:
        adjusted_neighbor = neighbor_idx
        for removed_idx in sorted(lr_indices):
            if removed_idx < neighbor_idx: adjusted_neighbor -= 1
        adjusted_connections.append(adjusted_neighbor)
    
    # --- For each connection point, add a copy of the arm ---
    for scaffold_connection_idx in adjusted_connections:
        # --- Process arm: remove one [Lr], keep one [Lr] ---
        arm_copy = Chem.RWMol(arm)
        # --- Find [Lr] anchors in arm ---
        arm_lr_indices = [a.GetIdx() for a in arm_copy.GetAtoms() if a.GetSymbol() == ANCHOR_SYMBOL]
        # --- Remove first [Lr] and connect to scaffold ---
        if arm_lr_indices:
            lr_to_remove = arm_lr_indices[0]
            lr_atom = arm_copy.GetAtomWithIdx(lr_to_remove)
            lr_neighbors = [n.GetIdx() for n in lr_atom.GetNeighbors()]
            
            if lr_neighbors:
                arm_connection_point = lr_neighbors[0]
                arm_copy.RemoveAtom(lr_to_remove)
                
                # --- Adjust arm connection point if needed ---
                if arm_connection_point > lr_to_remove:
                    arm_connection_point -= 1
                
                # --- Add arm to result ---
                arm_mapping = {}
                
                for i, atom in enumerate(arm_copy.GetAtoms()):
                    new_atom = Chem.Atom(atom.GetAtomicNum())
                    new_atom.SetFormalCharge(atom.GetFormalCharge())
                    new_idx = linker.AddAtom(new_atom)
                    arm_mapping[i] = new_idx
                
                # --- Add arm bonds ---
                for bond in arm_copy.GetBonds():
                    begin_new = arm_mapping[bond.GetBeginAtomIdx()]
                    end_new = arm_mapping[bond.GetEndAtomIdx()]
                    linker.AddBond(begin_new, end_new, bond.GetBondType())
                
                # --- Connect scaffold to arm by a single bond ---
                arm_connection_new = arm_mapping[arm_connection_point]
                linker.AddBond(scaffold_connection_idx, arm_connection_new, Chem.BondType.SINGLE)
    
    result = linker.GetMol()
    Chem.SanitizeMol(result)
    return result

# --- Symmetry-Constrained Design ---
class ScaffoldConstrainedDesign:
    def __init__(self,  model: VAEModel, tokenizer: Tokenizer, inference_config: dict, device: str = 'cpu'):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        self.tokenizer = tokenizer
        self.inference_config = inference_config
        self.pad_id = self.model.pad_index

    def design(self, scaffold: str, arm: str, samples: int = 2000, noise_sigma: int = 0.05):
        '''Sample the immediate neighborhood of a ditopic arm. Assemble new linkers.
        Return only the top-k samples based on fitness score.
        ---
        Args: - scaffold (str): Core scaffold (to be preserved).
              - arm (str): Ditopic arm to be used as a seed for sampling.
              - samples (int): Number of samples to generate. More samples can lead to better exploration at the cost of computational time
              - noise_sigma (float): Standard deviation of the Gaussian noise added to the latent variable. Higher the noise, more diverse the sampling
              - n_decodes (int): Number of decodes per latent variable to reduce noise
              - top_k (int): Number of top samples to return based on fitness score'''
        logger.info("Starting scaffold-constrained design")
        assert scaffold, "scaffold -> BAD"
        scaffold_mol = Chem.MolFromSmiles(scaffold)
        assert arm, "arm -> BAD"
        # --- Neighborhood sampling ---
        logger.info("Starting neighborhood sampling.")
        results = []
        # --- Encode seed ---
        tokens = self.tokenizer.encode_single(arm).unsqueeze(0).to(self.device)
        src_key_padding_mask = (tokens == self.pad_id).to(self.device)
        with torch.no_grad():
            z_seed, _, _ = self.model.encode(
                src=tokens, src_mask=None, src_key_padding_mask=src_key_padding_mask
            )
        z_seed = z_seed.squeeze(0)  # shape: [latent_dim]
        # --- Sample ---
        for _ in tqdm(range(samples), desc = "Neighborhood Sampling"):
            z = z_seed + noise_sigma * torch.randn_like(z_seed)
            z = z.unsqueeze(0).to(dtype=torch.float32).to(self.device)
            gen = self.model.generate(latent_variable = z, max_len=self.inference_config['max_len'],
                                temperature=self.inference_config['temperature'], top_p_val=self.inference_config['top_p_val'])
            smi = self.tokenizer.decode_single(gen[0])
            results.append(smi)
        # --- Assemble best linkers ---
        results = set(results)  # Remove duplicates
        logger.info(f"Assembling best linkers")
        generated_linkers = []
        for arm_smi in tqdm(results): 
            try:
                arm_connections = optimize_lr_placement(arm_smi, num_connections = 2)
                arm_mol = Chem.MolFromSmiles(arm_connections)
                linker = graft_arm_to_core_symmetric(scaffold_mol, arm_mol)
                linker_smi = Chem.MolToSmiles(linker, canonical=True)
                generated_linkers.append((arm_smi, linker_smi))
            except Exception: continue 
        logger.info(f"Generated {len(generated_linkers)} linkers after assembly")
        return generated_linkers
    
# ---- gating function(s) ----
'''Hard-chemistry filters -->
    1. Ring filters: remove molecules with 3- or 4-membered rings, or rings larger than 6
    2. Functional group filters: remove molecules with azides, diazo groups, nitroso groups, peroxides, sulfonates, phosphates, hydroxamic acids, carboxyl groups
    3. SCScore filter: remove molecules with SCScore above a threshold
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
        use_catalog = False; 
        print("Warning: GetFunctionalGroupHierarchy not available, using SMARTS patterns")
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

def scscore_filter(smiles: List[str], threshold: float = 3.5) -> List[str]:
    
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

# --- Load inference configurations ---
def load_config():
    '''Load inference configurations from a fixed (saved) file.'''
    config_path = '../../designed/linker/inference_config.txt'
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # --- Handle boolean values ---
                if value.lower() in ['true', 'false']:
                    config[key] = value.lower()
                elif value == 'auto':
                    config[key] = 'auto'
                elif '.' in value:
                    try:
                        config[key] = float(value)
                    except ValueError:
                        config[key] = value
                else:
                    try:
                        config[key] = int(value)
                    except ValueError:
                        config[key] = value
    return config

def auto_configs(config, dataset):
    '''Resolve auto configurations based on the dataset.'''
    if config['latent_dim'] == 'auto':
        config['latent_dim'] = dataset['latent_dim']
    if config['max_len'] == 'auto':
        config['max_len'] = dataset['max_len']
    return config

# --- Run ---
if __name__ == "__main__":
    display_banner()
    parser = argparse.ArgumentParser(description='Inference.')
    parser.add_argument('--mode', type = str, choices=['design', 'test'], default = 'design', help = 'design: scaffold-constrained design; test: unit tests for internal consistency')
    parser.add_argument('--delta', type = str, help = '0 - 0.1 for similar molecules, 0.1 - 0.2 for slightly diverse, 0.2+ for highly diverse')
    parser.add_argument('--filters', action = 'store_true', help = 'Apply hard-chemistry filters after generation')
    parser.add_argument('--reward', type = str, choices=['gas', 'grav'], default = 'gas', help = 'Reward function to use for reward filtering')
    parser.add_argument('--threshold', type = float, default = 0.5, help = 'Reward threshold for filtering')
    parser.add_argument('--output', type = str, default = '../../designed/linker/run/output.txt', help = 'Output file to save generated SMILES')
    args = parser.parse_args()

    # -----------------------------------------------------------
    # DO NOT CHANGE THIS BLOCK (unless you know what you're doing)
    # To avoid too many moving parts, all relative paths are fixed
    # All user-defined parameters go to the inference_config.txt file
    # -----------------------------------------------------------    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = '../../artifacts/ckpt/vae/no_prop_vae_epoch_120.pt'
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
    inference_config = load_config()
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
    try: checkpoint = torch.load(ckpt_path, map_location=device, weights_only = True)
    except TypeError: checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    logger.info("Model state loaded from %s", ckpt_path)

    # ---- Shared input/output setup ----
    run_dir = '../../designed/linker/run/'
    input = run_dir + 'input.txt'
    output = args.output
    assert os.path.isfile(input), f"Input file {input} does not exist. Please provide a valid input file."
    with open(input, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 2:
        raise ValueError("Input file must contain scaffold and arm SMILES on separate lines.")
    scaffold = lines[0]
    arm = lines[1]
    print(f"Input Scaffold: {scaffold}")
    print(f"Input Arm: {arm}")
    logger.info("Input configured, moving to inference.")

    def apply_filters(arm_linker_pairs):
        if args.filters:
            linkers = [linker for _, linker in arm_linker_pairs]
            ring_filtered = ring_filter(linkers, reject_three = True, reject_four = True)
            logger.info("%d molecules remain after ring filters.", len(ring_filtered))
            scs_filtered = scscore_filter(ring_filtered, threshold = 4.5)
            logger.info("%d molecules remain after SCScore filter.", len(scs_filtered))
            fg_filtered = fg_filter(scs_filtered, azide = True, diazo = True, nitroso = True, peroxide = True,
                                    sulfonate = True, phosphate = True, hydroxamic = True, carboxyl = True)
            logger.info("%d molecules remain after functional group filter.", len(fg_filtered))
            fg_filtered = list(set(fg_filtered))
            logger.info("%d UNIQUE molecules generated after [Lr] placement optimization & hard-chemistry filtering.", len(fg_filtered))
            logger.info("Moving to reward filtering.")
            return [(arm_smi, linker) for arm_smi, linker in arm_linker_pairs if linker in fg_filtered]
        logger.info("No filters applied. Moving directly to reward filtering.")
        return arm_linker_pairs

    def score_pairs(arm_linker_pairs):
        rf = RewardFunction()
        scored_pairs = []
        for arm_smi, linker in tqdm(arm_linker_pairs, desc = "Reward Scoring"):
            try:
                if args.reward == 'grav':
                    reward = rf.R_grav(linker)
                elif args.reward == 'gas':
                    reward = rf.R_gas(linker)
                else:
                    raise ValueError("Reward must be one of: gas, grav")
                if reward >= (args.threshold if args.threshold else 0.5):
                    scored_pairs.append((arm_smi, linker, reward))
            except Exception:
                logger.warning("Reward calculation failed for %s", linker); continue
        logger.info("%d UNIQUE molecules remain after reward filter", len(scored_pairs))
        scored_pairs.sort(key = lambda x: x[2], reverse = True)
        return scored_pairs

    # ---- Inference modes ----
    if args.mode == 'design':
        delta = float(args.delta) if args.delta else 0.1
        inference = ScaffoldConstrainedDesign(model = model, tokenizer = tokenizer, inference_config = inference_config, device = device)
        generated = inference.design(scaffold = scaffold, arm = arm, samples = inference_config['n_samples'],
                                     noise_sigma = delta)
        logger.info(f"Generated {len(generated)} linkers.")
        arm_linker_pairs = []
        num_connections = scaffold.count('Lr')
        for arm_smi, linker_smi in generated:
            try:
                opt_linker = optimize_lr_placement(linker_smi, num_connections = num_connections)
                arm_linker_pairs.append((arm_smi, opt_linker))
            except Exception as e:
                logger.warning("Failed to optimize [Lr] placement for %s: %s", linker_smi, str(e))
        arm_linker_pairs = list(set(arm_linker_pairs))
        scored_pairs = score_pairs(apply_filters(arm_linker_pairs))

    elif args.mode == 'test':
        logger.info("Running scaffold assembly test on the input scaffold/arm pair.")
        scaffold_mol = Chem.MolFromSmiles(scaffold)
        if scaffold_mol is None: raise ValueError("Invalid scaffold SMILES in input.txt")
        arm_connections = optimize_lr_placement(arm, num_connections = 2)
        arm_mol = Chem.MolFromSmiles(arm_connections)
        if arm_mol is None: raise ValueError("Invalid arm SMILES after connector placement")
        linker = graft_arm_to_core_symmetric(scaffold_mol, arm_mol)
        linker_smi = Chem.MolToSmiles(linker, canonical = True)
        num_connections = scaffold.count('Lr')
        opt_linker = optimize_lr_placement(linker_smi, num_connections = num_connections)
        arm_linker_pairs = [(arm, opt_linker)]
        logger.info("Constructed %d linker for test mode.", len(arm_linker_pairs))
        scored_pairs = score_pairs(apply_filters(arm_linker_pairs))

    final_pairs = [(arm_smi, linker) for arm_smi, linker, _ in scored_pairs]
    data = []
    for arm_smi, linker in final_pairs:
        data.append({
            'scaffold': scaffold,
            'arm': arm_smi,
            'linker': linker
        })
    df = pd.DataFrame(data)
    df.to_csv(output, index = False)

    logger.info("Generated SMILES saved to %s", output)
    logger.info("Stored in decreasing order of reward score.")
# ------------------------------------------------------------------
