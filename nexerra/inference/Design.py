# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  Naive inference on the trained molecular transformer
#  'Direct-design' mode 
#  Example usage:
#       python Design.py --mode neighborhood --delta [delta] \
#       --threshold [threshold] --filters True
#  Author: Dhruv Menon (dm958[at]cam[dot]ac[dot]uk)
#
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Standard libraries ---
import torch
import argparse
import pickle
import logging
import numpy as np
if not hasattr(np, 'bool'): np.bool = np.bool_

# --- logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from itertools import combinations
from collections import defaultdict, namedtuple
from tqdm import tqdm
from typing import List

# --- Local imports ---
from nexerra.model.HTVAE import VAEModel
from nexerra.utils.tokenizer import Tokenizer
from nexerra.inference.Reward import RewardFunction

import warnings
from botorch.exceptions.warnings import InputDataWarning
warnings.filterwarnings("ignore", category = InputDataWarning)

# --- RDKit ---
from rdkit import RDLogger
# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem.rdfiltercatalog import GetFunctionalGroupHierarchy

# --- SCScore ---
from nexerra.utils.scscore.scscore.standalone_model_numpy import SCScorer
logger.info("Synthetic complexity scorer loaded.")

import pyfiglet
def display_banner(): banner = pyfiglet.figlet_format("Nexerra", font = "slant"); print(banner)

# --- Neighborhood Sampling ---
class Neighborhood:
    def __init__(self, 
                 model: VAEModel, 
                 tokenizer: Tokenizer, 
                 inference_config: dict, 
                 device: str = 'cpu'):
        
        self.device = device
        self.model = model.to(self.device)
        self.model.eval() 
        self.tokenizer = tokenizer
        self.inference_config = inference_config
        self.fitness = RewardFunction()
        self.pad_id = self.model.pad_index

    def controlled_sample(self, 
                          seed: str, 
                          samples: int = 2000, 
                          noise_sigma: int = 0.05, 
                          top_k: int = 50):
        
        '''Sample the immediate neighborhood of a seed molecule. Return only the top-k samples based on fitness score.
        -------------------------------------------------------------
        Args:
        seed (str): Seed SMILES string.
        samples (int): Number of samples to generate. More samples can lead to better exploration at the cost of computational time.
        noise_sigma (float): Standard deviation of the Gaussian noise added to the latent variable. Higher the noise, more diverse the sampling.
        n_decodes (int): Number of decodes per latent variable to reduce noise.
        top_k (int): Number of top samples to return based on fitness score'''

        # --- Encode seed ---
        tokens = self.tokenizer.encode_single(seed).unsqueeze(0).to(self.device)
        src_key_padding_mask = (tokens == self.pad_id).to(self.device)
        with torch.no_grad():
            z_seed, _, _ = self.model.encode(
                src=tokens, src_mask=None, src_key_padding_mask=src_key_padding_mask)
        z_seed = z_seed.squeeze(0)  # shape: [latent_dim]
        best_score = 0.0
        results = []
        for _ in tqdm(range(samples), desc = "Neighborhood Sampling"):
            z = z_seed + noise_sigma * torch.randn_like(z_seed)
            z = z.unsqueeze(0).to(dtype = torch.float32).to(self.device)
            gen = self.model.generate(latent_variable = z, max_len = self.inference_config['max_len'],
                                    temperature = self.inference_config['temperature'], top_p_val = self.inference_config['top_p_val'])
            smi = self.tokenizer.decode_single(gen[0])
            try: score = self.fitness.R_grav(smi)
            except Exception as e: print(f"Error calculating fitness for {smi}: {e}"); continue
            if not np.isnan(score):
                results.append((smi, score))
                if score > best_score: best_score = score

        results.sort(key = lambda x: x[1], reverse = True)
        return results[:top_k]  # Return only the top-k samples based on fitness score

# --------------------------------
#          Hard-chemistry filters
# These filters are applied post-generation to filter out strained, poor quality molecules.
# We look at (i) molecules containing non-5/6-membered rings,
# (ii) reactive functional groups, and high ring strain,
# (iii) we will also reject molecules with triple bonds in rings.
# --------------------------------

def ring_filter(smiles: List[str], reject_three: bool = True, reject_four: bool = True, reject_large: bool = True) -> List[str]:
    '''Filter generated SMILES based on the number of rings [...]
    Also removes molecules with triple bonded carbons in rings (unable to form conformers)'''
    
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
        has_ring_triple_bond = False
        for b in mol.GetBonds():
            if b.GetBondType() == Chem.BondType.TRIPLE and b.IsInRing(): 
                has_ring_triple_bond = True; break
        if not has_ring_triple_bond: filtered.append(smi)
    return filtered

def fg_filter(smiles: List[str], azide: bool = True, diazo: bool = True, nitroso: bool = True, peroxide: bool = True,
              sulfonate: bool = True, phosphate: bool = True, hydroxamic: bool = True, carboxyl: bool = True) -> List[str]:
    
    '''Avoid unstable/reactive functional groups <-- destabilize the linker.
    Filters out azides, diazo groups, nitroso groups, and peroxides (by default),
    but can be toggled based on the needs'''
    
    # --- Set up the functional group filter ---
    _HAZARD_PREFIXES = []
    if azide == True: _HAZARD_PREFIXES.append("Azide")
    if diazo == True: _HAZARD_PREFIXES.append("Diazo")
    if nitroso == True: _HAZARD_PREFIXES.append("Nitroso")
    if peroxide == True: _HAZARD_PREFIXES.append("Peroxide")
    if sulfonate: _HAZARD_PREFIXES.append("Sulfonate")
    if phosphate: _HAZARD_PREFIXES.append("Phosphate")
    if hydroxamic: _HAZARD_PREFIXES.extend(["Hydroxamic", "Hydroxylamino"])
    if carboxyl: _HAZARD_PREFIXES.extend(["Carboxyl", "Carboxylic"])
    
    # --- Set up the filter catalog ---
    try:
        _FGCAT = GetFunctionalGroupHierarchy(); use_catalog = True
    except Exception:
        use_catalog = False; print("Warning: GetFunctionalGroupHierarchy not available, using SMARTS patterns")
    
    # --- define SMARTS patterns for additional functional groups ---
    # why? cause these may conflict with place -[Lr]; logic can be finetuned 
    additional_smarts = {}
    if sulfonate: additional_smarts['sulfonate'] = ["S(=O)(=O)O"]
    if phosphate: additional_smarts['phosphate'] = ["P(=O)(O)(O)O"]  
    if hydroxamic: additional_smarts['hydroxamic'] = ['[CX3](=O)N[OX2H1]']
    if carboxyl: additional_smarts['carboxylic'] = ['C(=O)O', 'C(=O)[OH]', '[CX3](=O)[OH]']
    
    # --- filter based on hazardous functional groups ---
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
            except Exception:
                pass
        
        # --- Switch to the SMARTS pattern otherwise ---
        if not has_hazard:
            for _, patterns in additional_smarts.items():
                for pattern in patterns:
                    try:
                        pattern_mol = Chem.MolFromSmarts(pattern)
                        if pattern_mol is not None and mol.HasSubstructMatch(pattern_mol):
                                has_hazard = True; break
                    except Exception: continue
                if has_hazard: break
        # --- If no hazardous groups, keep the molecule ---
        if not has_hazard: filtered_smiles.append(smi)
    return filtered_smiles

def scscore_filter(smiles: List[str], threshold: float = 4.0) -> List[str]:
    '''Filter generated SMILES based on the SCScore'''
    # --- Initialize SCScore model ---
    scmodel = SCScorer()
    scmodel.restore(os.path.join('../utils','scscore', 'models', 'full_reaxys_model_2048bool', 'model.ckpt-10654.as_numpy.json.gz'), FP_rad = 2, FP_len = 2048)
    # --- Filter based on SCScore ---
    filtered = []
    for smi in smiles:
        clean_smi = smi.replace("[Lr]", "*")
        mol = Chem.MolFromSmiles(clean_smi)
        if mol is None: continue
        _, sco = scmodel.get_score_from_smi(clean_smi)
        if sco <= threshold: filtered.append(smi)
    return filtered

# -------------------------------------
# Helper functions for connector [Lr] placement 
# Core logic has been described in the paper [...]
# -------------------------------------

def prepare_molecule(smi):
    '''To optimise the placement of [Lr] connectors, we need to prepare the molecule.'''
    mol = Chem.MolFromSmiles(smi)
    if mol is None: raise ValueError("Invalid SMILES")

    rw_mol = Chem.RWMol(mol)
    lr_indices = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == "Lr"]

    for idx in sorted(lr_indices, reverse = True):
        atom = rw_mol.GetAtomWithIdx(idx)
        if atom.IsInRing(): atom.SetAtomicNum(7) # If [Lr] is in a ring, replace it with a nitrogen atom
        else: rw_mol.RemoveAtom(idx) # If [Lr] is not in a ring, remove it
        
    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol

def get_candidate_sites(mol):
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    if AllChem.EmbedMolecule(mol, randomSeed = 42) != 0: AllChem.Compute2DCoords(mol)

    conf = mol.GetConformer()
    candidates = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C': continue
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'H':
                idx = atom.GetIdx()
                pos = conf.GetAtomPosition(idx)
                candidates.append({'idx': idx, 'coords': np.array([pos.x, pos.y, pos.z])})
                break
    return candidates

def placement_score(coords):
    dists = [np.linalg.norm(coords[i] - coords[j]) for i in range(len(coords)) for j in range(i+1, len(coords))]
    mean_dist = np.mean(dists)
    return mean_dist

def select_best_sites(candidates, num_connections):
    best_score = -np.inf
    best_combo = None
    for combo in combinations(candidates, num_connections):
        coords = [c['coords'] for c in combo]
        score = placement_score(coords)
        if score > best_score: best_score = score; best_combo = combo
    return [c['idx'] for c in best_combo]

def place_lr_atoms(mol, selected_indices):
    rw_mol = Chem.RWMol(mol)
    for idx in selected_indices:
        atom = rw_mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'H':
                rw_mol.RemoveAtom(neighbor.GetIdx())
                break
        lr_atom = Chem.Atom("Lr")
        new_idx = rw_mol.AddAtom(lr_atom)
        rw_mol.AddBond(idx, new_idx, Chem.BondType.SINGLE)
    Chem.SanitizeMol(rw_mol)
    return rw_mol.GetMol()

def optimize_lr_placement(smi, num_connections = 4):
    mol = prepare_molecule(smi)
    candidates = get_candidate_sites(mol)
    if len(candidates) < num_connections: raise ValueError("Not enough candidate sites for desired connections")
    selected = select_best_sites(candidates, num_connections)
    mol_with_lr = place_lr_atoms(mol, selected)
    return Chem.MolToSmiles(mol_with_lr)


# --------------------------------------------------
# For inference, we use configuration files saved at FIXED paths.
# This is done to help potential users who are not much familiar with coding.
# This can ofcourse be edited [...]
# --------------------------------------------------

# --- Load inference configurations ---
def load_config():
    '''Load inference configurations from a fixed (saved) file'''
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


if __name__ == "__main__":
    display_banner()
    parser = argparse.ArgumentParser(description='Inference.')
    parser.add_argument('--mode', type = str, choices=['neighborhood', 'connectors'], default = 'neighborhood', help='Mode: Design or connector optimisation')
    parser.add_argument('--delta', type = str, help = '0 - 0.05 for similar molecules, 0.05 - 0.1 for slightly diverse, 0.1 - 0.2 for highly diverse')
    parser.add_argument('--threshold', type = float, default = 1.0, help = 'Threshold multiplier for selecting better molecules than the seed (default: 1.0).')
    parser.add_argument('--filters', type = bool, default = True, help = 'Apply hard chemistry filters post-generation (default: True)')
    parser.add_argument('--num_connections', type = int, default = 4, help = 'Number of [Lr] connections to place in the scaffold (default: 4).')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------
    # We used fixed paths for the model for ease of use, do not change unless you know where it is.
    # -------------

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
    
    try: checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError: checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint['model_state'])
    logger.info("Model state loaded from %s", ckpt_path)

    # ---- Inference modes ----
    if args.mode == 'neighborhood':
        # ----- Load and configure the input seed -----
        run_dir = '../../designed/linker/run/'
        input = run_dir + 'input.txt'
        output = run_dir + 'output.txt'
        assert os.path.isfile(input), f"Input file {input} does not exist. Please provide a valid input file."
        if input.endswith('.txt'): 
            with open(input, 'r') as f: input = f.read().strip()
        else: input = input.strip()
        print(f"Input SMILES: {input}")
        logger.info("Input configured, moving to inference.")
        num_connections = input.count('[Lr]')
        assert num_connections > 1, "Input SMILES must contain at least two connections."
        # ----- Inference loop -----
        delta = float(args.delta) if args.delta else 0.1
        inference = Neighborhood(model = model, tokenizer = tokenizer, inference_config = inference_config, device = device)
        results = inference.controlled_sample(seed = input, samples = inference_config['n_samples'],
                                              noise_sigma = delta, top_k = inference_config['top_k'])
        logger.info(f"Generated {len(results)} SMILES.")
        reward = RewardFunction()
        best_smi = []
        # -------------------------
        # The reward functions are currently hard-coded; this can be modularised better. earmarked future work.
        # -------------------------
        seed_score = reward.R_grav(smi = input)
        threshold = args.threshold if args.threshold else 0.9
        for smi, _ in results:
            try:
                score = reward.R_grav(smi = smi)
                if score > threshold * seed_score: best_smi.append(smi)
            except Exception as e:
                logger.error(f"Error calculating reward for SMILES {smi}: {e}")
        best_smi = list(set(best_smi))  # Remove duplicates
        logger.info(f"Filtered {len(best_smi)} SMILES that outperform seed linker by {threshold}x.")
        if args.filters == True:
            logger.info("Applying chemistry filters to generated SMILES")
            best_smi_ring = ring_filter(best_smi, reject_three = True, reject_four = True, reject_large = True)
            best_smi_fg = fg_filter(best_smi_ring, azide = True, diazo = True, nitroso = True, peroxide = True,
                                    sulfonate = True, phosphate = True, hydroxamic = True, carboxyl = True)
            best_smi = scscore_filter(best_smi_fg, threshold = 4.25)
            logger.info(f"{len(best_smi)} SMILES remain after applying chemistry filters.")
        if best_smi:
            logger.info(f"Preparing MOF linkers")
            top_linkers_connections = []
            for linker in best_smi:
                try:
                    linker_connections = optimize_lr_placement(linker, num_connections = num_connections)
                    logger.info(f"Linker --> {linker_connections}")
                    logger.info(f"Score --> {reward.R_grav(linker)}")
                    top_linkers_connections.append(linker_connections)
                except Exception as e:
                    logger.error(f"Error processing linker {linker}: {e}")
            output_linkers = output.replace('.txt', '_linkers.txt')
            with open(output_linkers, 'w') as f:
                for linker in top_linkers_connections: f.write(f"{linker}\n")
            logger.info(f"Results saved to {output} & {output_linkers}.")
        else: logger.warning("No suitable linkers generated - maybe reduce your threshold.")

    if args.mode == 'connectors':
        # ----- Load and configure the input seed -----
        input_smiles = []
        run_dir = '../../designed/run/'
        input = run_dir + 'input.txt'
        output = run_dir + 'output.txt'
        assert os.path.isfile(input), f"Input file {input} does not exist. Please provide a valid input file."
        with open(input, 'r') as f:
            lines = f.readlines()
            for line in lines:
                smi = line.strip()
                if smi: input_smiles.append(smi)
        
        # ----- Load reward function -----
        linkers = []
        for smi in input_smiles:
            assert smi, "SMILES string is empty."
            num_connections = args.num_connections if args.num_connections else 4
            linker = optimize_lr_placement(smi, num_connections = num_connections)
            linkers.append(linker)
            
        with open(output, 'w') as f:
            for linker in linkers:
                f.write(f"{linker}\n")
        logger.info(f"Generated {len(linkers)} linkers. Results saved to {output}.")


