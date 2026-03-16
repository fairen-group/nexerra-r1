# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Diagnostics for the training dataset.
# Edited & modified by: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License
#  Copyright (c) 2025 Dhruv Menon
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
# ---------------------------------------

import os, sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse, random, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----- RDKit imports -----
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize as rms
RDLogger.DisableLog('rdApp.*')

import pyfiglet
def display_banner(): banner = pyfiglet.figlet_format("Nexerra", font="slant"); print(banner)

# --- Standardization pipeline ---
def standardize_mol(mol, do_standardize = True):
    '''Return standardized mol; original multi-fragment flag'''
    if mol is None: return None, False
    if not do_standardize:
        # --- Still detect multi-fragments on the raw mol ---
        multifrag = (len(Chem.GetMolFrags(mol)) > 1)
        try: Chem.SanitizeMol(mol)
        except Exception: return None, multifrag
        return mol, multifrag

    try:
        # --- cleanup: [normalize + reionize + sanity] --- 
        params = rms.CleanupParameters()
        mol = rms.Cleanup(mol, params)
        # --- Multi-fragment detection BEFORE choosing largest fragment ---
        multifrag = (len(Chem.GetMolFrags(mol)) > 1)
        # --- Keep largest fragment ---
        lfc = rms.LargestFragmentChooser()
        mol = lfc.choose(mol)
        # --- uncharge ---
        uncharger = rms.Uncharger()
        mol = uncharger.uncharge(mol)
        # --- canonical tautomer ---
        te = rms.TautomerEnumerator()
        mol = te.Canonicalize(mol)
        Chem.SanitizeMol(mol)
        return mol, multifrag
    except Exception: return None, False

# ----- fingerprints for diversity -----
def ecfp4_bitvect(mol, radius = 2, nBits = 2048):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = radius, nBits = nBits)

def tanimoto_sim(fp1, fp2):
    from rdkit.DataStructs import TanimotoSimilarity
    return TanimotoSimilarity(fp1, fp2)

# ----- Internal diversity (sampled) -----
def estimate_internal_diversity(mols, n_pairs = 200000, rng = None):
    '''Return IntDiv = 1 - mean Tanimoto similarity for sampled pairs'''
    if rng is None: rng = random.Random(0)
    if len(mols) < 2: return float('nan')
    # --- Precompute ECFP4 for all sampled mols ---
    fps = [ecfp4_bitvect(m) for m in mols]
    n = len(fps)
    n_pairs = min(n_pairs, n*(n-1)//2)
    sims = []
    for _ in range(n_pairs):
        i = rng.randrange(n); j = rng.randrange(n-1)
        if j >= i: j += 1
        sims.append(tanimoto_sim(fps[i], fps[j]))
    sims = np.array(sims, dtype=np.float32)
    return float(1.0 - sims.mean())

# ----- descriptors -----
def basic_descriptors(mol):
    '''Return a dict of simple physchem descriptors'''
    return {
        'MW': Descriptors.MolWt(mol),
        'MolLogP': Crippen.MolLogP(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
    }

def bemis_murcko_scaffold_smi(mol):
    try:
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaff, canonical = True) if scaff is not None else None
    except Exception: return None

def run(args):
    t0 = time.time()
    rng = random.Random(args.seed)
    total = 0; parsed = 0; standardized = 0; multifrag_count = 0
    inchikeys = set()
    dup_count = 0
    desc = defaultdict(list)
    scaff_counter = Counter()

    # --- For diversity sampling we keep (mol, smi) to avoid recomputing ---
    sample_reservoir = []

    def row_iter():
        for chunk in pd.read_csv(args.csv, chunksize = args.chunksize):
            for smi in chunk[args.smiles_column].astype(str): yield smi

    for smi in tqdm(row_iter(), desc = "Processing SMILES"):
        total += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        parsed += 1

        mol_std, multifrag = standardize_mol(mol, do_standardize = not args.no_standardize)
        if mol_std is None: continue
        standardized += 1
        if multifrag: multifrag_count += 1

        # --- identity after standardization: use InChIKey ---
        try: ik = Chem.InchiToInchiKey(Chem.MolToInchi(mol_std))
        except Exception: ik = Chem.MolToSmiles(mol_std, canonical = True)
        if ik in inchikeys: dup_count += 1
        else: inchikeys.add(ik)

        # --- Descriptors ---
        d = basic_descriptors(mol_std)
        for k, v in d.items():
            desc[k].append(v)

        # --- Scaffold stats ---
        scaff_smi = bemis_murcko_scaffold_smi(mol_std)
        if scaff_smi: scaff_counter[scaff_smi] += 1

        # --- Reservoir sample for diversity ---
        if len(sample_reservoir) < args.sample_size: sample_reservoir.append((mol_std, ik))
        else:
            j = rng.randint(1, standardized)
            if j <= args.sample_size: sample_reservoir[j-1] = (mol_std, ik)

    # --- Diversity on sample ---
    sampled_mols = [m for (m, _) in sample_reservoir]
    intdiv = estimate_internal_diversity(sampled_mols, n_pairs = args.pairs, rng = rng)

    # --- Summaries ---
    def summarize(x):
        x = np.array(x, dtype = float)
        if len(x) == 0: return {}
        return {
            'count': len(x),
            'mean': float(np.mean(x)),
            'std': float(np.std(x)),
            'min': float(np.min(x)),
            'p10': float(np.percentile(x, 10)),
            'p25': float(np.percentile(x, 25)),
            'median': float(np.median(x)),
            'p75': float(np.percentile(x, 75)),
            'p90': float(np.percentile(x, 90)),
            'max': float(np.max(x)),
        }

    summaries = {k: summarize(v) for k, v in desc.items()}

    # --- Scaffold diversity ---
    n_scaff = len(scaff_counter)
    top_scaff = scaff_counter.most_common(10)
    top_scaff_rel = [(s, c, c / max(1, standardized)) for (s, c) in top_scaff]

    # --- Write diagnostics ---
    with open(args.out, 'w', encoding='utf-8') as f:
        w = f.write
        w("=== DATASET DIAGNOSTICS ===\n")
        w(f"Input CSV : {os.path.abspath(args.csv)}\n")
        w(f"SMILES col: {args.smiles_column}\n")
        w(f"Standardize: {not args.no_standardize} (MolStandardize available: YES)\n")
        w(f"Sample size for diversity: {args.sample_size}, pairs: {args.pairs}\n")
        w("\n--- Sizes & Hygiene ---\n")
        w(f"Total rows                   : {total}\n")
        w(f"Parsed (RDKit MolFromSmiles) : {parsed} ({parsed/total*100:.2f}%)\n")
        w(f"Standardized + sanitized     : {standardized} ({(standardized/total*100):.2f}%)\n")
        w(f"Multi-fragment (before LFC)  : {multifrag_count} ({(multifrag_count/max(1,parsed))*100:.2f}%)\n")
        w(f"Unique identities (InChIKey) : {len(inchikeys)}\n")
        w(f"Exact duplicates (post-std)  : {dup_count} ({(dup_count/max(1,standardized))*100:.2f}%)\n")

        w("\n--- Descriptor Distributions ---\n")
        for k in ["MW", "MolLogP", "TPSA", "HBA", "HBD", "Rings"]:
            s = summaries.get(k, {})
            w(f"[{k}]\n")
            for kk in ["count","mean","std","min","p10","p25","median","p75","p90","max"]:
                if kk in s: w(f"  {kk:>6}: {s[kk]:.4f}\n")
            w("\n")

        w("--- Diversity (sample-based) ---\n")
        w(f"Sampled molecules for diversity    : {len(sampled_mols)}\n")
        w(f"Internal diversity (ECFP4 IntDiv)  : {intdiv:.4f}  # = 1 - mean(Tanimoto)\n")

        w("\n--- Scaffold Diversity ---\n")
        w(f"Unique Bemis–Murcko scaffolds      : {n_scaff}\n")
        w("Top-10 scaffolds (SMILES, count, fraction of standardized):\n")
        for s, c, rel in top_scaff_rel:
            w(f"  {s}\t{c}\t{rel:.6f}\n")

        w("\n--- Runtime ---\n")
        w(f"Wall time (s): {time.time()-t0:.2f}\n")

        w("\n--- Notes ---\n")
        w("* Standardization uses RDKit MolStandardize (cleanup → largest fragment → uncharge → canonical tautomer) when available.\n")
        w("* MolLogP is Crippen's RDKit implementation; TPSA/HBA/HBD/Rings via RDKit descriptors.\n")
        w("* Internal diversity computed on a random reservoir sample with random fingerprint pairs (ECFP4, 2048 bits).\n")
        w("* Adjust --sample_size/--pairs/--chunksize for speed/precision trade-offs.\n")

def main():
    p = argparse.ArgumentParser(description="Dataset QA for SMILES CSV (writes diagnostics.txt)")
    p.add_argument("--csv", required=True, help="Path to CSV containing SMILES")
    p.add_argument("--smiles_column", default="smiles", help="Column name with SMILES (default: smiles)")
    p.add_argument("--out", default="diagnostics.txt", help="Output diagnostics file")
    p.add_argument("--chunksize", type=int, default=100000, help="CSV chunk size (default: 100k)")
    p.add_argument("--sample_size", type=int, default=10000, help="Reservoir sample size for diversity")
    p.add_argument("--pairs", type=int, default=200000, help="Random pairs for IntDiv")
    p.add_argument("--no_standardize", action="store_true", help="Skip MolStandardize (faster)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    display_banner(); main()
