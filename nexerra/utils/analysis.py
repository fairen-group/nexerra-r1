# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Dataset analysis script
# Computes: 
#           - heavy atom counts
#           - molecular weight
#           - logP
#           - rotatable bonds
#           - heteroatom distribution
#           - SCScore
#           - Bemis-Murcko scaffolds
#           - scaffold statistics
#           - pairwise Tanimoto similarity distribution
#           - UMAP embedding coloured by scaffold
# Usage:
#    python analysis.py --input input.csv --col smiles --out analysis_out
# Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
from collections import Counter
import numpy as np
if not hasattr(np, 'bool'): np.bool = np.bool_
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# --- RDKit ---
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
# Silence RDKit warnings
RDLogger.DisableLog("rdApp.*")
# --- UMAP ---
from umap import umap_ as umap
# --- SCScore ---
from nexerra.utils.scscore.scscore.standalone_model_numpy import SCScorer
print("Synthetic complexity scorer loaded.")

def parse_args():
    parser = argparse.ArgumentParser(description = "Dataset analysis")
    parser.add_argument("--input", type = str, help = "Input dataset")
    parser.add_argument("--col", type = str, default = "smiles", help = "Name of the SMILES column in the CSV (default: smiles)")
    parser.add_argument("--out", type = str, default = "analysis", help = "Output directory for .txt files and plots")
    parser.add_argument("--fp-bits", type = int, default = 2048, help = "Number of bits for Morgan fingerprints (default: 2048)")
    parser.add_argument("--fp-radius", type = int, default = 2, help = "Radius for Morgan fingerprints (default: 2 = ECFP4)",)
    parser.add_argument("--n-sim-pairs", type = int, default = 200000, help = "Number of random fingerprint pairs for similarity distribution",)
    parser.add_argument("--umap-sample-size", type = int, default = 50000, help = "Number of molecules to sample for UMAP (default: 50k)",)
    parser.add_argument("--random-seed", type = int, default = 42,)
    return parser.parse_args()

def load_smiles(input_csv, smiles_col):
    df = pd.read_csv(input_csv)
    smiles = df[smiles_col].astype(str).tolist()
    return smiles

def init_scscore():
    scmodel = SCScorer()
    model_path = os.path.join(
        os.path.dirname(__file__),
        "scscore",
        "models",
        "full_reaxys_model_2048bool",
        "model.ckpt-10654.as_numpy.json.gz",
    )
    scmodel.restore(model_path, FP_rad = 2, FP_len = 2048)
    return scmodel

def compute_basic_descriptors(smiles, fp_bits = 2048, fp_radius = 2, sc_model = None):
    heavy_atoms = []
    mol_weights = []
    logp = []
    rot_bonds = []
    hetero_per_mol = []  
    scaffolds = []
    fps = []
    scscores = []

    total_hetero_counts = Counter()

    n = len(smiles)
    for _, smi in tqdm(enumerate(smiles), total = n, desc = "Computing descriptors"):
        smi = smi.replace('[Lr]', '*')
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            heavy_atoms.append(np.nan)
            mol_weights.append(np.nan)
            logp.append(np.nan)
            rot_bonds.append(np.nan)
            hetero_per_mol.append({})
            scaffolds.append("NA")
            fps.append(None)
            scscores.append(np.nan)
            continue

        heavy_atoms.append(mol.GetNumHeavyAtoms())
        mol_weights.append(Descriptors.MolWt(mol))
        logp.append(Crippen.MolLogP(mol))
        rot_bonds.append(Lipinski.NumRotatableBonds(mol))

        hetero_counts = Counter()
        for atom in mol.GetAtoms():
            z = atom.GetAtomicNum()
            if z not in (1, 6):
                sym = atom.GetSymbol()
                hetero_counts[sym] += 1
                total_hetero_counts[sym] += 1
        hetero_per_mol.append(dict(hetero_counts))

        try:
            scaff_mol = MurckoScaffold.GetScaffoldForMol(mol)
            if scaff_mol is None or scaff_mol.GetNumAtoms() == 0: scaff_smiles = "NA"
            else: scaff_smiles = Chem.MolToSmiles(scaff_mol, isomericSmiles = False)
        except Exception:
            scaff_smiles = "NA"
        scaffolds.append(scaff_smiles)

        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = fp_radius, nBits = fp_bits)
        fps.append(fp)

        try:
            _, score = sc_model.get_score_from_smi(smi)
            scscores.append(float(score))
        except Exception:
            scscores.append(np.nan)
            
    results = {
        "heavy_atoms": heavy_atoms,
        "mol_weights": mol_weights,
        "logp": logp,
        "rot_bonds": rot_bonds,
        "hetero_per_mol": hetero_per_mol,
        "total_hetero_counts": total_hetero_counts,
        "scaffolds": scaffolds,
        "fps": fps,
        "scscores": scscores,
    }
    return fps, results


def write_txt(values, out_path, header = None):
    with open(out_path, "w") as f:
        if header is not None: f.write("# " + header + "\n")
        for v in values:
            if v is None or (isinstance(v, float) and np.isnan(v)): f.write("NA\n")
            else: f.write(f"{v}\n")


def write_heteroatoms_txt(hetero_per_mol, total_hetero_counts, out_path):
    '''Writes heteroatom distribution to text file:
    - overall counts at top
    - per-molecule counts as tab-separated table'''

    all_syms = sorted(total_hetero_counts.keys())

    with open(out_path, "w") as f:
        f.write("# --- Overall heteroatom counts across dataset:\n")
        for sym in all_syms: f.write(f"# {sym}: {total_hetero_counts[sym]}\n")
        f.write("#\n")
        f.write("# Per-molecule heteroatom counts (one row per molecule)\n")
        f.write("index\t" + "\t".join(all_syms) + "\ttotal\n")

        for idx, hdict in enumerate(hetero_per_mol):
            if not hdict:
                row = ["0"] * len(all_syms)
                total = 0
            else:
                row = [str(hdict.get(sym, 0)) for sym in all_syms]
                total = sum(hdict.values())
            f.write(f"{idx}\t" + "\t".join(row) + f"\t{total}\n")


def write_scaffolds_txt(scaffolds, out_path_per_mol, out_path_stats):
    with open(out_path_per_mol, "w") as f:
        f.write("# Bemis-Murcko scaffold SMILES per molecule (aligned with input order)\n")
        for s in scaffolds: f.write(f"{s}\n")

    counts = Counter(scaffolds)
    if "NA" in counts: del counts["NA"]

    total = sum(counts.values())
    unique = len(counts)

    with open(out_path_stats, "w") as f:
        f.write("# Scaffold statistics\n")
        f.write(f"total_scaffold_assignments\t{total}\n")
        f.write(f"unique_scaffolds\t{unique}\n")
        f.write("\n# Top 50 scaffolds (scaffold_smiles\tcount\tfraction)\n")

        for scaff, cnt in counts.most_common(50):
            frac = cnt / total if total > 0 else 0.0
            f.write(f"{scaff}\t{cnt}\t{frac:.6f}\n")

def compute_similarity_samples(fps, n_pairs, rng):
    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid_indices) < 2: return np.array([])

    sims = []
    for _ in tqdm(range(n_pairs), desc = "Sampling similarities"):
        i, j = rng.choice(valid_indices, size = 2, replace = False)
        fp_i = fps[i]
        fp_j = fps[j]
        sim = DataStructs.TanimotoSimilarity(fp_i, fp_j)
        sims.append(sim)
    return np.array(sims)

def compute_nn(fps, rng, sample_size = 20000):
    '''Approximate nearest-neighbour Tanimoto similarity for a random subset of molecules'''
    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid_indices) < 2: return np.array([]), []
    sample_size = min(sample_size, len(valid_indices))
    sample_indices = rng.choice(valid_indices, size = sample_size, replace=False)
    sub_fps = [fps[i] for i in sample_indices]
    n = len(sub_fps)
    nn_sims = np.zeros(n, dtype=float)
    # For each sampled molecule, compute similarity to all others in the subset
    for i, fp_i in tqdm(enumerate(sub_fps), desc = 'Computing nearest-neighbour similarities'):
        sims = DataStructs.BulkTanimotoSimilarity(fp_i, sub_fps)
        sims[i] = -1.0
        nn_sims[i] = max(sims)
    return nn_sims, list(sample_indices)

def write_similarity_stats(sims, out_samples, out_stats):
    with open(out_samples, "w") as f:
        f.write("# Sampled Tanimoto similarities\n")
        for s in sims: f.write(f"{s:.6f}\n")

    with open(out_stats, "w") as f:
        f.write("# Similarity statistics\n")
        if sims.size == 0:
            f.write("no_valid_pairs\t0\n")
            return

        f.write(f"n_samples\t{sims.size}\n")
        f.write(f"mean\t{float(np.mean(sims)):.6f}\n")
        f.write(f"median\t{float(np.median(sims)):.6f}\n")
        for p in [5, 25, 75, 95]:
            val = float(np.percentile(sims, p))
            f.write(f"p{p}\t{val:.6f}\n")


def run_umap_and_plot(fps, scaffolds, sample_size, rng, out_png):
    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    if len(valid_indices) == 0:
        print("No valid fingerprints for UMAP.", file=sys.stderr)
        return

    sample_size = min(sample_size, len(valid_indices))
    sample_indices = rng.choice(valid_indices, size=sample_size, replace=False)

    n_bits = fps[valid_indices[0]].GetNumBits()
    X = np.zeros((sample_size, n_bits), dtype=np.float32)
    for k, idx in enumerate(sample_indices):
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fps[idx], arr)
        X[k, :] = arr

    sampled_scaffolds = [scaffolds[idx] for idx in sample_indices]
    counts = Counter(s for s in sampled_scaffolds if s != "NA")
    top_scaffolds = {s for s, _ in counts.most_common(15)}

    labels = []
    for s in sampled_scaffolds:
        if s in top_scaffolds:
            labels.append(s)
        else: labels.append("Other")

    uniq_labels = sorted(set(labels))
    label_to_int = {lab: i for i, lab in enumerate(uniq_labels)}
    y = np.array([label_to_int[lab] for lab in labels], dtype=int)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="jaccard",
        random_state=int(rng.integers(0, 1_000_000)),
    )
    embedding = reducer.fit_transform(X)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y, s=3, alpha=0.7,)
    handles = []
    for lab, idx in label_to_int.items():
        handles.append(
            plt.Line2D(
                [], [], marker="o", linestyle="none", label=lab,
                markersize=5
            )
        )
    plt.legend(
        handles,
        label_to_int.keys(),
        fontsize=6,
        loc="best",
        markerscale=1.5,
    )
    plt.title("UMAP of linkers (Morgan fingerprints) coloured by scaffold")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved UMAP plot to {out_png}", file=sys.stderr)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok = True)

    print("Loading SMILES...", file=sys.stderr)
    smiles = load_smiles(args.input, args.col)
    print(f"Loaded {len(smiles)} SMILES.", file=sys.stderr)

    rng = np.random.default_rng(args.random_seed)

    print("Initialising SCScore model", file = sys.stderr)
    sc_model = init_scscore()

    print("Computing descriptors, scaffolds and fingerprints...", file=sys.stderr)
    fps, results = compute_basic_descriptors(
        smiles,
        fp_bits=args.fp_bits,
        fp_radius=args.fp_radius,
        sc_model=sc_model,
    )

    write_txt(
        results["heavy_atoms"],
        os.path.join(args.out, "heavy_atoms.txt"),
        header="Heavy atom count per molecule (aligned with input order)",)
    
    write_txt(
        results["mol_weights"],
        os.path.join(args.out, "molecular_weight.txt"),
        header="Molecular weight (RDKit MolWt) per molecule",)
    
    write_txt(
        results["logp"],
        os.path.join(args.out, "logP.txt"),
        header="Crippen MolLogP per molecule",)
    
    write_txt(
        results["rot_bonds"],
        os.path.join(args.out, "rotatable_bonds.txt"),
        header="Number of rotatable bonds per molecule (Lipinski.NumRotatableBonds)",)
    
    write_txt(
        results["scscores"],
        os.path.join(args.out, "scscore.txt"),
        header="SCScore per molecule (NA if unavailable or invalid)",)

    # Heteroatoms
    write_heteroatoms_txt(
        results["hetero_per_mol"],
        results["total_hetero_counts"],
        os.path.join(args.out, "heteroatoms.txt"),)

    # Scaffolds
    write_scaffolds_txt(
        results["scaffolds"],
        os.path.join(args.out, "scaffolds_per_mol.txt"),
        os.path.join(args.out, "scaffold_stats.txt"),)

    # Similarity sampling
    print("Sampling pairwise Tanimoto similarities...", file=sys.stderr)
    sims = compute_similarity_samples(
        results["fps"],
        n_pairs=args.n_sim_pairs,
        rng=rng,
    )
    write_similarity_stats(
        sims,
        os.path.join(args.out, "similarity_samples.txt"),
        os.path.join(args.out, "similarity_stats.txt"),
    )

    # Nearest-neighbour similarities
    print("Computing nearest-neighbour Tanimoto similarities...", file=sys.stderr)
    nn_sims, _ = compute_nn(
        fps,
        rng,
        sample_size = 50000,
    )

    write_similarity_stats(
        nn_sims,
        os.path.join(args.out, "nn_similarity_samples.txt"),
        os.path.join(args.out, "nn_similarity_stats.txt"),
    )

    # UMAP
    print("Running UMAP and plotting...", file=sys.stderr)
    run_umap_and_plot(
        results["fps"],
        results["scaffolds"],
        sample_size=args.umap_sample_size,
        rng=rng,
        out_png=os.path.join(args.out, "umap_scaffold.png"),
    )

    print("Done.", file = sys.stderr)


if __name__ == "__main__": main()
