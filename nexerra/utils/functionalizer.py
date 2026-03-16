# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Rapid functionalization of SMILES.
# Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import random
random.seed(42)  # For reproducibility
import argparse
import numpy as np
if not hasattr(np, 'bool'): np.bool = np.bool_
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(handler)

# --- RDKit ---
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit import RDLogger
# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# --- SCScore ---
from scscore.scscore.standalone_model_numpy import SCScorer
logger.info("Synthetic complexity scorer loaded.")

# --- Allowed functional groups (customize as necessary) ---
functional_group_list = [
    'F', 'Cl', 'Br', 'I',
    'O', 'S', 'N',
    'C', 'CC', 'CCC', 'CCCC',
    'c1ccccc1', 'c1ccncc1',
    'C=O', 'C(=O)O', 'C#N', 
    'C(F)(F)F',
    'C1CCO1', 'C1CCN1'] 

# --- utility function(s) ---
def valid_molecule(mol: Chem.Mol) -> bool:
    '''Check if the molecule is valid and does not contain any invalid atoms'''
    try: rdmolops.SanitizeMol(mol); return True
    except: return False    

# --- Functionalization step ---
def functionalize_step(mol: Chem.Mol, fg_smi: str) -> Chem.Mol:
    '''Random monovalent functionalization
    Based on the passed fragment SMILES'''
    mol_h = Chem.AddHs(mol)
    rw = Chem.RWMol(mol_h)
    replaceable_atoms = []
    for atom in rw.GetAtoms():
        if atom.GetSymbol() in ['H', 'Lr']: continue
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == 'H':
                replaceable_atoms.append(atom.GetIdx())
                break        
    if not replaceable_atoms: return None
    
    target_index = random.choice(replaceable_atoms)
    for nbr in rw.GetAtomWithIdx(target_index).GetNeighbors():
        if nbr.GetSymbol() == 'H':
            rw.RemoveAtom(nbr.GetIdx())
            break
    
    fg_mol = Chem.MolFromSmiles(fg_smi)
    if fg_mol is None: return None
    
    combined = Chem.CombineMols(rw.GetMol(), fg_mol)
    rw2 = Chem.RWMol(combined)
    new_atom_index = rw.GetNumAtoms()
    rw2.AddBond(target_index, new_atom_index, Chem.BondType.SINGLE)
    new_mol = rw2.GetMol()
    Chem.SanitizeMol(new_mol)
    if valid_molecule(new_mol) == False: return None
    return new_mol
        
# --- main functionalization logic ---
def random_functionalization_n_constrained(model, smiles: str, max_runs: int = 3, max_attempts: int = 10) -> str:
    '''Between 1 and max_runs functionalization steps. Ensure that the synthetic complexity score is below 4.75.'''

    orig_mol = Chem.MolFromSmiles(smiles)
    if orig_mol is None: return None
    for _ in range(max_attempts):
        mol = Chem.Mol(orig_mol) # reset to the original molecule if the previous attempt is rejected
        runs = random.randint(1, max_runs)
        for _ in range(runs):
            fg = random.choice(functional_group_list)
            augmented = functionalize_step(mol, fg)
            if augmented is None: break
            mol = Chem.RemoveHs(augmented)

        smi = Chem.MolToSmiles(mol, canonical=True)
        _, sco = model.get_score_from_smi(smi)
        
        if np.isnan(sco): continue
        if sco < 4.75: return smi

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Functionalization')
    parser.add_argument('--mode', type = str, required = True, choices = ['data', 'sample'], help = 'data for entire dataset, sample for testing a single SMILES')
    parser.add_argument('--src', type = str, help = 'Path to the training dataset')
    parser.add_argument('--dst', type = str, help = 'Path to save the augmented dataset')
    parser.add_argument('--variants', type = int, help = 'Number of variants to generate')
    parser.add_argument('--max_attempts', type = int, help = 'Number of attempts to generate a valid molecule')
    parser.add_argument('--sample', type = str, help = 'Sample SMILES for testing')
    args = parser.parse_args()

    if args.mode == 'data':
        assert args.src is not None and args.dst is not None, "Source and destination paths must be provided"
        src = args.src
        dst = args.dst
        variants = args.variants if args.variants is not None else 3
        attempts = args.max_attempts if args.max_attempts is not None else 10

        orig_dataset = pd.read_csv(src)
        logger.info(f"Loaded dataset with {len(orig_dataset)} SMILES from {src}")
        smiles = orig_dataset['smiles'].tolist()
        unique_smiles = list(set(smiles))
        augmented_smiles = []
        
        model = SCScorer()
        model.restore(os.path.join('scscore', 'models', 'full_reaxys_model_2048bool', 'model.ckpt-10654.as_numpy.json.gz'), FP_rad = 2, FP_len = 2048)

        logger.info(f"Starting functionalization")
        for smi in tqdm(unique_smiles):
            for _ in range(variants):
                aug_smi = random_functionalization_n_constrained(model, smi, 3, attempts)
                if aug_smi is not None:
                    augmented_smiles.append(aug_smi)
        
        # filter out exact duplicates (NOT canonical duplicates) to reduce overheads
        augmented_smiles = list(set(augmented_smiles))
        augmented_smiles = [smi for smi in augmented_smiles if smi not in unique_smiles] # remove original SMILES
        augmented_len = len(augmented_smiles)
        logger.info(f"Original dataset size: {len(unique_smiles)}")
        logger.info(f"Augmented dataset size: {augmented_len}")
        augmented_dataset = pd.DataFrame(augmented_smiles, columns=['smiles'])
        augmented_dataset.to_csv(dst, index=False)

    elif args.mode == 'sample':        
        sample_smiles = args.sample if args.sample is not None else "[Lr]c2ccc(c1ccc([Lr])cc1)cc2"
        augmented = []
        variants = args.variants if args.variants is not None else 3
        attempts = args.max_attempts if args.max_attempts is not None else 10
        model = SCScorer()
        model.restore(os.path.join('scscore', 'models', 'full_reaxys_model_2048bool', 'model.ckpt-10654.as_numpy.json.gz'), FP_rad = 2, FP_len = 2048)
        for _ in range(variants):
            augmented.append(random_functionalization_n_constrained(model, sample_smiles, 3, attempts))
        logger.info(f"Original: {sample_smiles}")
        for smi in augmented:
            logger.info(f"Augmented: {smi}")