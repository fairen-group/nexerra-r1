# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  Reward function definitions for linker design. 
#  Includes a general-purpose reward function for gas storage applications, 
#       see bio <-- for some preliminary work on drug delivery
#  can be adapted for specific applications in a [plug-and-play] manner.
#  Author: Dhruv Menon (dm958[at]cam[dot]ac[dot]uk); Drug delivery function formulated by Ivan Zyuzin
#
#  MIT License
#  Copyright (c) 2026 Dhruv Menon
# ---------------------------------------

from __future__ import annotations
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import csv
import numpy as np
if not hasattr(np, 'bool'): np.bool = np.bool_ 
import random; random.seed(42)
import math
from math import exp
from tqdm import tqdm
import argparse
from itertools import combinations
from typing import List, Tuple

# --- Logging ---
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- RDKit ---
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, DataStructs, Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdmolops import AddHs

# --- SCScore ---
'''Ensure the SCScore module is available in the path (else, reconfigure)'''
from nexerra.utils.scscore.scscore.standalone_model_numpy import SCScorer
logger.info("Synthetic complexity scorer loaded")

# --- Banner ---
import pyfiglet
def display_banner(): banner = pyfiglet.figlet_format("Nexerra", font="slant"); print(banner)

# --- Filter warnings ---
import warnings
from botorch.exceptions.warnings import InputDataWarning
with warnings.catch_warnings(): warnings.simplefilter("ignore", category = InputDataWarning)
    
# --- Reward Function Class ---
'''The RewardFunction class encapsulates various reward components for linker 
design. In principle, the reward functions can be adapted for different applications 
and can be made arbitrarily complex - as long as the principles discussed in 
the manuscript are adhered to. Here, we outline the reward function for the case studies
relevant to the manuscript'''

# -----------------------------
# GAS STORAGE REWARD FUNCTION
# -----------------------------
class RewardFunction:
    def __init__(self):
        self.scmodel = SCScorer()
        current = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(current, '..', 'utils', 'scscore', 'models', 'full_reaxys_model_2048bool', 'model.ckpt-10654.as_numpy.json.gz')
        weight_path = os.path.abspath(weight_path)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"SCScore weights not found at {weight_path}")
        self.scmodel.restore(weight_path, FP_rad = 2, FP_len = 2048)

    '''To make the reward scores more smooth, we apply smoothing functions;
    Depending on the property of interest, we either use sigmoid functions or gaussian functions'''    
    def _bounded(self, y0: float, lo: float = 0.05, hi: float = 0.95) -> float:
        '''Bound the output to [lo, hi]; here [0.05, 0.95] 
        Prevents extreme values from distorting landscape'''
        y0 = float(max(0.0, min(1.0, y0)))
        return lo + (hi - lo) * y0
    def gaussian_desirability(self, x: float, center: float, width: float, lo: float = 0.05, hi: float = 0.95) -> float:
        y = math.exp(-0.5 * ((float(x) - float(center)) / float(width)) ** 2)
        return self._bounded(y, lo = lo, hi = hi)
    def decreasing_sigmoid_desirability(self, x: float, center: float, width: float, lo: float = 0.05, hi: float = 0.95) -> float:
        sigma_dec = 1.0 / (1.0 + math.exp((float(x) - float(center)) / float(width)))
        return self._bounded(sigma_dec, lo = lo, hi = hi)
    def increasing_sigmoid_desirability(self, x: float, center: float, width: float, lo: float = 0.05, hi: float = 0.95) -> float:
        sigma_inc = 1.0 / (1.0 + math.exp(-(float(x) - float(center)) / float(width)))
        return self._bounded(sigma_inc, lo = lo, hi = hi)
    
    # --- Anchor-to-anchor length ---
    '''A 3D-embedding-based method; fallback to a graph-based method [...]
    This however, can fail for highly complex/weird linkers
    Expect a 2 - 3% failure rate (on training set), perhaps more on generated molecules'''
    
    def _replace_Lr(self, mol: Chem.Mol, anchor_symbol: str = "Lr") -> Tuple[Chem.Mol, List[int]]:
        '''Anchor - anchor length depends on 3D embedding, which could give issues if the [Lr] anchor is present
        replace each [Lr] with a chemically reasonable benign Carbon
            - convert anchor --> carbon [in place, i.e., indices preserved];
        Returns
        ---
            - new_mol
            - anchor_idxs: indices where anchors were found (now carbon)'''
        
        rw = Chem.RWMol(mol)
        anchor_idxs = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == anchor_symbol]
        if len(anchor_idxs) < 2: raise ValueError(f"Expected atleast 2 anchors [{anchor_symbol}] but found {len(anchor_idxs)}")

        # convert anchors into carbons so RDKit embedding doesnt fail
        for idx in anchor_idxs:
            a = rw.GetAtomWithIdx(idx)
            a.SetAtomicNum(6)  
            a.SetFormalCharge(0)
            a.SetNoImplicit(False)
            a.SetIsotope(0)

        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol, anchor_idxs
    
    def _anchor_distance_embed(self, smi: str, anchor_symbol: str = "Lr",
        seed: int = 42, do_uff_opt: bool = False, prune_rms_thresh: float = 0.5) -> Tuple[float, List[int]]:
        '''Compute anchor-to-anchor distance via ETKDG embedding after replacing [Lr] with C
        Return max pairwise anchor distance (agnostic of number of anchors - >= 2)
        Returns --> distance or np.nan if embedding fails'''

        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Invalid SMILES: {smi}"

        mol2, anchor_idxs = self._replace_Lr(mol, anchor_symbol = anchor_symbol)
        mol3 = Chem.AddHs(mol2)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.pruneRmsThresh = float(prune_rms_thresh)

        if AllChem.EmbedMolecule(mol3, params) != 0:
            return float(np.nan), anchor_idxs

        if do_uff_opt:
            try: AllChem.UFFOptimizeMolecule(mol3, maxIters = 200)
            except Exception: pass

        conf = mol3.GetConformer()
        coords = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol3.GetNumAtoms())], dtype = float)

        dmax = 0.0
        for i in range(len(anchor_idxs)):
            for j in range(i + 1, len(anchor_idxs)):
                d = float(np.linalg.norm(coords[anchor_idxs[i]] - coords[anchor_idxs[j]]))
                if d > dmax: dmax = d
        return float(dmax) if dmax > 0 else float(np.nan), anchor_idxs

    def _graph_anchor_distance(self, mol, lr_indices, bond_len: float = 1.45) -> float:
        # This is by all means, a very crude proxy [...]
        if len(lr_indices) < 2: return 0.0
        max_d = 0.0
        for i, j in combinations(lr_indices, 2):
            path = Chem.GetShortestPath(mol, i, j)
            num_bonds = max(0, len(path) - 1) 
            max_d = max(max_d, bond_len * num_bonds)
        return float(max_d)

    def anc_to_anc_length(self, smi: str) -> float:
        # Much cleaner, yet embedding can fail for the complex ones [...]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Invalid SMILES string: {smi}"
        anchor_distance, _ = self._anchor_distance_embed(smi, anchor_symbol = "Lr", seed = 42, do_uff_opt = True, prune_rms_thresh = 0.5)
        if anchor_distance == 0.0 or np.isnan(anchor_distance):
            try:
                lr_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Lr"] 
                anchor_distance = self._graph_anchor_distance(mol, lr_indices, bond_len = 1.45)
            except Exception: pass
        return anchor_distance if anchor_distance > 0 else np.nan
    
    def desirability_end_to_end_length(self, length: float, center: float = 16.0, width: float = 6.0) -> float:
        return self.increasing_sigmoid_desirability(length, center = center, width = width, lo = 0.05, hi = 0.95)
    # ------------------------------------------
   
    # --- Flexibility proxy (rotatable bond fraction + fraction of CSP3) ---
    def flex_score(self, mol) -> float:
        '''continuous flexibility in [0,1] --> blend of rotatable bond fraction and FractionCSP3'''
        try: rot = float(rdMolDescriptors.CalcNumRotatableBonds(mol, strict = True))
        except Exception: rot = 0.0
        nb = max(1, mol.GetNumBonds())
        rot_frac = max(0.0, min(1.0, rot / nb))
        try: csp3 = float(Descriptors.FractionCSP3(mol))
        except Exception: csp3 = 0.0
        flex = 0.5 * rot_frac + 0.5 * csp3
        return float(max(0.0, min(1.0, flex)))

    def desirability_flex(self, flex: float, center: float = 0.5, width: float = 0.2) -> float:
        return self.decreasing_sigmoid_desirability(flex, center = center, width = width, lo = 0.05, hi = 0.95)
    # ------------------------------------------

    # --- Molecular Weight ---
    '''The molecular weight of the linker is an important consideration for volumetric
    gas storage applications. Higher molecular weights need to be penalized'''

    def molecular_weight(self, smi: str) -> float:
        '''clip off the [Lr] anchors for MW calculation
        Since all linkers essentially have a terminal group, this should effectively cancel out on comparison'''
        try:
            mol = Chem.MolFromSmiles(smi)
            lr_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'Lr']
            rw = Chem.RWMol(mol)
            for idx in sorted(lr_idxs, reverse = True): rw.RemoveAtom(idx)
            clean_mol = rw.GetMol()
            assert clean_mol is not None, f"Invalid SMILES string: {smi}"
            Chem.SanitizeMol(clean_mol)
            molwt = rdMolDescriptors.CalcExactMolWt(clean_mol)
        except Exception: molwt = np.nan
        return molwt
    
    def desirability_molwt(self, mw: float, center: float = 320.0, width: float = 140.0) -> float:
        return self.gaussian_desirability(mw, center = center, width = width, lo = 0.05, hi = 0.95)
    # ------------------------------------------

    # --- Mass-normalized-span ---
    def mass_normalized_span(self, smi: str) -> float:
        '''Anchor-to-anchor length normalized by molecular weight'''
        try:
            length = self.anc_to_anc_length(smi)
            mw = self.molecular_weight(smi)
            if np.isnan(length) or np.isnan(mw) or mw == 0.0: return float(np.nan)
            return float(length) / (max(float(mw), 1e-4)) ** float(0.8)
        except Exception: return float(np.nan)

    def desirability_spm(self, spm: float, center: float = 0.03, width: float = 0.005) -> float:
        '''Higher mass-normalized-span is better (more span per unit mass) -> increasing sigmoid
        Tune center/width from dataset quantiles'''
        if spm is None or np.isnan(spm): return float(np.nan)
        return self.increasing_sigmoid_desirability(spm, center = center, width = width, lo = 0.05, hi = 0.95)
    
    # --- Symmetry (proxies) ---
    '''For linker symmetry, we consider the similarity (Tanimoto, Morgan Fp) between the environments
    of the anchor atoms[...] The environments are defined as the k-hop neighborhoods around each anchor
    ---> A higher similarity index indicates a more symmetric environment around the anchors
    [Note, this is by all means yet another proxy; RDKIT doesnt have a built-in-implementation as far as I'm aware]'''
    
    def _k_hop_env_atoms(self, mol, center_idx, hops = 3):
        # --- standard BFS to get k-hop neighborhood atoms ---
        seen, frontier = {center_idx}, {center_idx}
        for _ in range(hops):
            nxt = set()
            for a in frontier: nxt.update(n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors())
            frontier = nxt - seen
            seen |= frontier
        return list(seen)
    
    def anchor_env_symmetry_agnostic(self, smi: str, fp_radius = 3, nBits = 2048, env_hops = 3, agg = "softmin") -> float:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Invalid SMILES string: {smi}"
        anchors = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Lr"]
        assert len(anchors) > 1, "Expected at least two anchor atoms (Lr) in the molecule"  
        fps = []
        for idx in anchors:
            atoms = self._k_hop_env_atoms(mol, idx, env_hops) # get k-hop environment(s)
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = fp_radius, nBits = nBits, fromAtoms = atoms)
            fps.append(fp)
        # --- pairwise similarities b/w fingerprints --- 
        sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i in range(len(fps)) for j in range(i+1, len(fps))]
        if not sims: return 0.0
        mean_s = float(sum(sims)) / len(sims)
        if agg == "mean": return mean_s
        if agg == "min": return min(sims)
        # (default) softmin = mean - std
        var = sum((s - mean_s) ** 2 for s in sims) / len(sims)
        std = math.sqrt(var)
        return max(0.0, min(1.0, mean_s - std))

    def desirability_anchor_env_symmetry(self, smi: str) -> float:
        val = self.anchor_env_symmetry_agnostic(smi, env_hops = 3, agg = "softmin")
        return max(0.05, min(0.95, float(val))) 
    # ------------------------------------------
    
    # --- Composite reward score ---
    ''' R_grav: a1 * len + a2 * Flexibility + a3 * SPM
        R_gas: b1 * Length + b2 * Flexibility + b3 * Symmetry + b4 * MW'''
    
    def R_grav(self, smi: str) -> float:
        '''Base reward (gravimetric) with continuous components, scaled to [0.5, 0.95]'''
        assert smi is not None, "SMILES string is None"
        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol, isomericSmiles = True, canonical = True) 
        len = self.anc_to_anc_length(smi)
        d_len = self.desirability_end_to_end_length(len)
        spm = self.mass_normalized_span(smi)
        flex = self.flex_score(mol)
        d_spm = self.desirability_spm(spm)        
        d_flex = self.desirability_flex(flex)
        return float(0.4 * d_len + 0.4 * d_flex + 0.2 * d_spm)
    
    def R_gas(self, smi: str) -> float:
        '''Length + Flex + Symm + MW, scaled to [0.5, 0.95]'''
        assert smi is not None, "SMILES string is None"
        assert '[Lr]' in smi, "SMILES string does not contain anchor atoms [Lr]"
        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol, isomericSmiles = True, canonical = True) 
        len = self.anc_to_anc_length(smi)
        if len is None or np.isnan(len): return float(np.nan)
        d_len = self.desirability_end_to_end_length(len)
        flex = self.flex_score(mol)
        d_flex = self.desirability_flex(flex)
        d_symm = self.desirability_anchor_env_symmetry(smi)
        mw = self.molecular_weight(smi)
        d_mw = self.desirability_molwt(mw)
        # --- weights: 0.3 (length) + 0.2 (flex) + 0.3 (symm) + 0.2 (mw) [CAN BE TUNED] --- 
        return float(0.20 * d_len + 0.20 * d_flex + 0.40 * d_symm + 0.20 * d_mw)

if __name__ == "__main__":
    display_banner()
    parser = argparse.ArgumentParser(description = 'Reward Functions')
    parser.add_argument('--mode', type = str, required = True, choices = ['distribution', 'benchmark'], help='distribution: calculate & save distributions of reward scores \
                        \n benchmark: test reward function on a set of linkers - to tune properties')
    parser.add_argument('--data', type = str, help = 'Path to the data file containing SMILES strings for analysing distributions')
    parser.add_argument('--samples', type = int, default = 1000, help = 'Number of samples to draw for distribution calculation (if applicable)')
    parser.add_argument('--smi', type = str, help = 'For testing, provide a path to .txt file containing the MOF linkers.')
    args = parser.parse_args()

    if args.mode == 'distribution': 
        if not args.data: raise ValueError("Please provide a data file path using --data")
        with open(args.data, 'r') as f: linkers = [line.strip() for line in f if line.strip()]
        rf = RewardFunction()
        grav_scores = []
        gas_scores = []
        sampled = linkers if len(linkers) <= args.samples else random.sample(linkers, args.samples)
        for smi in tqdm(sampled, desc = 'Calculating the reward distribution'):
            grav_sco = rf.R_grav(smi)
            if not np.isnan(grav_sco): grav_scores.append((smi, grav_sco))
            gas_sco = rf.R_gas(smi)
            if not np.isnan(gas_sco): gas_scores.append((smi, gas_sco))
        grav_output = 'grav_scores.csv'
        with open(grav_output, 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(['SMILES', 'R_grav'])
            for smi, sco in grav_scores: writer.writerow([smi, f"{sco:.4f}"])
        logger.info(f"Saved reward scores to {grav_output}")

        gas_output = 'gas_scores.csv'
        with open(gas_output, 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(['SMILES', 'R_gas'])
            for smi, sco in gas_scores: writer.writerow([smi, f"{sco:.4f}"])
        logger.info(f"Saved reward scores to {gas_output}")

    if args.mode == 'benchmark':
        if not os.path.exists(args.smi): raise FileNotFoundError(f"text file not found: {args.smi}")
        linkers = []
        with open(args.smi, 'r') as f:
            for line in f:
                smi = line.strip() 
                if smi: linkers.append(smi)
        rf = RewardFunction()
        scores = []
        logger.info(f"Calculating rewards for {len(linkers)} linker(s) [...]")
        for smi in linkers:
            score = rf.R_grav(smi); scores.append((smi, score))
            logger.info(f"SMILES: {smi} --> R_grav: {score:.4f}")
    
