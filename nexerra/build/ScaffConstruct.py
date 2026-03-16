# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  Batch assembly of MOFs using generated linkers & pre-defined nets via ToBaCCo3.0
#  For complex linkers, we have one inorganic node and an organic node, along with an edge.
#  We will process the organic edge and node --> assemble with TobaCCo3.0 on the specified net.
#  Example usage: 
#  python ScaffConstruct.py --input [path_to_pairs.csv] --tobacco-dir [path_to_tobacco3.0] 
#  --net [net: eg. pcu] --output-dir [output_cif_dir]
#  Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

# ---------------------------------------
# Several notes here:
# 1. Please bring your own TobaCCo(3.0)
# 2. This script can ONLY be automated for a particular topology-type
# 3. For construction related issues, refer to the parent publications.
# 4. Since our current work focuses more on linker design, completed automated MOF construction remains outside the scope
# For single node, single edge cases, use Construct.py
# Credits for the TobaCCo3.0 code: "Cryst. Growth Des. 2017, 17, 11, 5801–5810"
# Place Tobacco3.0 in this directory, name the folder 'tobacco3.0'
# ---------------------------------------

'''admittedly, this code is a bit clunky, can be improved'''

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import re
import math
import shutil
import pandas as pd
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Set

# --- RDKit ---
from rdkit import Chem
from rdkit.Chem import AllChem

import pyfiglet
def display_banner(): banner = pyfiglet.figlet_format("Nexerra", font="slant"); print(banner)

# --- Helper functions ---
def _sanitize_for_filename(text: str, max_len: int = 120) -> str:
    s = re.sub(r"\s+", "_", text.strip())
    s = re.sub(r"[^A-Za-z0-9._+=-]", "-", s)
    return s[:max_len] if max_len else s

def _read_smiles_lines(path: str) -> List[str]:
    '''Read SMILES from .txt/.csv file,
    For .txt: Ensure one SMILES per line
    For .csv: Ensure SMILES is first column
    '''
    lines: List[str] = []
    with open(path, "r") as f:
        for raw in f:
            t = raw.strip()
            if not t or t.startswith("#"): continue
            if "," in t: t = t.split(",", 1)[0].strip()
            lines.append(t)
    return lines

# --- Write CIFs for ToBaCCo edges ---
def write_cif_from_atoms(atoms: List, cif_path: str, box_size: float = 20.0) -> None:
    if not atoms: raise ValueError("No atoms provided to write CIF")
    coords = [(x, y, z) for (_, x, y, z) in atoms]
    cx = sum(x for x, _, _ in coords) / len(coords)
    cy = sum(y for _, y, _ in coords) / len(coords)
    cz = sum(z for _, _, z in coords) / len(coords)
    # Convert to fractional coordinates in box
    frac = [((x - cx) / box_size + 0.5, (y - cy) / box_size + 0.5, (z - cz) / box_size + 0.5) for (x, y, z) in coords]

    with open(cif_path, "w") as w:
        w.write("data_building_block\n")
        w.write(f"_cell_length_a {box_size:.6f}\n")
        w.write(f"_cell_length_b {box_size:.6f}\n")
        w.write(f"_cell_length_c {box_size:.6f}\n")
        w.write("_cell_angle_alpha 90.0\n")
        w.write("_cell_angle_beta 90.0\n")
        w.write("_cell_angle_gamma 90.0\n")
        w.write("loop_\n")
        w.write("_atom_site_label\n")
        w.write("_atom_site_type_symbol\n")
        w.write("_atom_site_fract_x\n")
        w.write("_atom_site_fract_y\n")
        w.write("_atom_site_fract_z\n")
        labels: List[str] = []

        # --- For ToBaCCo, connectors are labeled 'X' ---
        def is_connector_symbol(s: str) -> bool:
            us = s.upper(); return s == "[Lr]" or us == "X" or us == "V"

        '''Use sequential numbering based on atom position so labels run 1[...]N
           eg. C1, X2, C3, X4, H5 ... This avoids per-element counters that produce
          numbering like C1, X1, C2, X2, H1 which can confuse downstream code'''
        
        for idx, (sym, _, _, _) in enumerate(atoms, start = 1):
            x, y, z = frac[idx - 1]
            if is_connector_symbol(sym):
                label = f"X{idx}"
                labels.append(label)
                # Keep connector recorded as a C-type in the CIF (consistent with ToBaCCo expectations)
                w.write(f"{label} C {x:.6f} {y:.6f} {z:.6f}\n")
            else:
                esym = sym
                label = f"{esym}{idx}"
                labels.append(label)
                w.write(f"{label} {esym} {x:.6f} {y:.6f} {z:.6f}\n")

        if len(labels) >= 2:
            carts = [(fx * box_size, fy * box_size, fz * box_size) for (fx, fy, fz) in frac]
            def dist(i: int, j: int) -> float:
                xi, yi, zi = carts[i]
                xj, yj, zj = carts[j]
                return math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)

            visited = {0}
            edges = []
            while len(visited) < len(labels):
                best = (1e18, None, None)
                for i in list(visited):
                    for j in range(len(labels)):
                        if j in visited: continue
                        d = dist(i, j)
                        if d < best[0]: best = (d, i, j)
                if best[1] is None: break
                edges.append((best[1], best[2], best[0]))
                visited.add(best[2])

            w.write("loop_\n")
            w.write("_geom_bond_atom_site_label_1\n")
            w.write("_geom_bond_atom_site_label_2\n")
            w.write("_geom_bond_distance\n")
            w.write("_geom_bond_site_symmetry_2\n")
            w.write("_ccdc_geom_bond_type\n")
            for (i, j, d) in edges: w.write(f"{labels[i]} {labels[j]} {d:.3f} . S\n")

def generate_node_atoms_from_smiles(smiles: str, num_confs: int = 20, seed: int = 42) -> List[Tuple[str, float, float, float]]:
    '''Generate 3D conformer from linker SMILES specifically for ToBaCCo3.0 nodes
    ---
    Places connection points close to the molecular structure for proper bonding'''
    # Use Iodine as temporary placeholder to avoid UFF dummy atom errors
    smi_temp = smiles.replace("[Lr]", "[I]")
    mol = Chem.MolFromSmiles(smi_temp)
    if mol is None: raise ValueError("Failed to parse SMILES (after [Lr]->[I] replacement).")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs = max(1, num_confs), params = params))
    
    try: AllChem.UFFOptimizeMoleculeConf(mol, confId = conf_ids[0])
    except Exception: pass

    anchor_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 53]
    body_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 53]
    
    if len(anchor_idxs) < 1 or not body_idxs: 
        raise ValueError("SMILES must contain at least one [Lr] anchor and other atoms[...]")

    conf = mol.GetConformer(conf_ids[0])
    
    # For nodes: Place connection points close to the molecular structure
    '''im not very happy with this approach, but struggled to otherwise resolve problems'''
    atoms: List[Tuple[str, float, float, float]] = []
    
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        if a.GetAtomicNum() == 53:  # Skip anchors for now
            continue
            
        pos = conf.GetAtomPosition(idx)
        sym = a.GetSymbol()
        atoms.append((sym, float(pos.x), float(pos.y), float(pos.z)))
    
    # place connection points close to the anchor positions
    for ai in anchor_idxs:
        anchor_atom = mol.GetAtomWithIdx(ai)
        anchor_pos = conf.GetAtomPosition(ai)
        
        # Find the neighbor of the anchor atom (the atom it's bonded to)
        neighbors = [b.GetOtherAtomIdx(ai) for b in anchor_atom.GetBonds()]
        
        if neighbors:
            # Get the position of the first neighbor
            neighbor_idx = neighbors[0]
            neighbor_pos = conf.GetAtomPosition(neighbor_idx)
            
            # Calculate direction from neighbor to anchor
            dx = anchor_pos.x - neighbor_pos.x
            dy = anchor_pos.y - neighbor_pos.y
            dz = anchor_pos.z - neighbor_pos.z
            
            # Normalize the direction vector
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            if length > 0:
                dx /= length; dy /= length; dz /= length
                
                # Place connection point just slightly beyond the neighbor atom
                bond_distance = 0.0
                conn_x = neighbor_pos.x + dx * bond_distance
                conn_y = neighbor_pos.y + dy * bond_distance
                conn_z = neighbor_pos.z + dz * bond_distance
            else:
                # Fallback -- place at anchor position
                conn_x = anchor_pos.x
                conn_y = anchor_pos.y
                conn_z = anchor_pos.z
        else:
            # No neighbors found - place at anchor position
            conn_x = anchor_pos.x
            conn_y = anchor_pos.y
            conn_z = anchor_pos.z
        
        atoms.append(('V', float(conn_x), float(conn_y), float(conn_z)))  # Use 'V' for node connections    
    return atoms


def generate_edge_atoms_from_smiles(smiles: str, num_confs: int = 20, seed: int = 42) -> List[Tuple[str, float, float, float]]:
    '''Generate 3D conformer from linker SMILES for ToBaCCo3.0 edges
    --- avoids UFF dummy atom issues by using temporary placeholders'''
    # Use Iodine as temporary placeholder to avoid UFF dummy atom errors
    smi_temp = smiles.replace("[Lr]", "[I]")
    mol = Chem.MolFromSmiles(smi_temp)
    if mol is None: raise ValueError("Failed to parse SMILES (after [Lr]->[I] replacement).")
    
    # mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs = max(1, num_confs), params = params))
    
    # UFF optimization with real atoms (no dummy atom issues)
    try: AllChem.UFFOptimizeMoleculeConfs(mol, confIds = conf_ids)
    except Exception: pass

    anchor_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 53]
    body_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 53]
    
    if len(anchor_idxs) < 1 or not body_idxs: 
        raise ValueError("SMILES must contain at least one [Lr] anchor and other atoms.")

    conf = mol.GetConformer(conf_ids[0])
    connector_idxs: List[int] = []
    drop_anchor: Set[int] = set()
    keep_anchor_as_x: Set[int] = set()
    
    for ai in anchor_idxs:
        a = mol.GetAtomWithIdx(ai)
        if a.IsInRing(): 
            keep_anchor_as_x.add(ai)
            connector_idxs.append(ai)
        else:
            nbrs = [b.GetOtherAtomIdx(ai) for b in a.GetBonds()]
            if len(nbrs) != 1: 
                raise ValueError(f"Non-ring anchor at idx {ai} has {len(nbrs)} neighbors; expected exactly 1.")
            ni = nbrs[0]
            connector_idxs.append(ni)
            drop_anchor.add(ai)

    atoms: List[Tuple[str, float, float, float]] = []
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        if idx in drop_anchor: continue
        pos = conf.GetAtomPosition(idx)
        sym = 'X' if idx in connector_idxs else a.GetSymbol()
        atoms.append((sym, float(pos.x), float(pos.y), float(pos.z)))
    
    return atoms

def run_tobacco(tobacco_dir: str, python_exe: Optional[str] = None) -> None:
    tob_dir = Path(tobacco_dir).resolve()
    if not tob_dir.exists():
        alt = (Path(__file__).resolve().parent / 'tobacco3.0').resolve()
        if alt.exists(): tob_dir = alt
        else: raise FileNotFoundError(f"tobacco_dir not found: {tobacco_dir}")
    script = 'tobacco.py'
    if not (tob_dir / script).exists():
        raise FileNotFoundError(f"Expected {script} in {tob_dir}, but it was not found")

    py = python_exe
    if py and Path(py).name.lower() == 'tobacco.py':
        print("[note] --tobacco-py expects a Python interpreter (e.g., /path/to/venv/bin/python); ignoring passed tobacco.py")
        py = None
    if not py:
        py = sys.executable or 'python'

    proc = subprocess.run([py, str(script)], cwd = str(tob_dir), capture_output = True, text = True)
    if proc.stdout:
        print("--- ToBaCCo stdout ---")
        print(proc.stdout)
    if proc.stderr:
        print("--- ToBaCCo stderr ---")
        print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"ToBaCCo failed with exit code {proc.returncode}")


def main(argv: Optional[List[str]] = None) -> int:
    display_banner()
    ap = argparse.ArgumentParser(description = "Organic node + edge --> ToBaCCo --> output .cif pipeline")
    ap.add_argument('--input', required = True, help = 'Path to CSV with scaffold and arm columns')
    ap.add_argument('--tobacco-dir', required = True, help = 'Path to tobacco3.0 directory')
    ap.add_argument('--net', required = True, help = 'Net name (must exist in template_database/<net>.cif)')
    ap.add_argument('--output-dir', required = True, help = 'Directory to place produced .cif files')
    ap.add_argument('--tobacco-py', default = None, help = 'Python executable to run tobacco.py (defaults to current env)')
    ap.add_argument('--node-cif', default = None, help = 'Optional path to node CIF to copy once to nodes/input_node.cif')
    args = ap.parse_args(argv)

    tob_dir = Path(args.tobacco_dir).resolve()
    if not tob_dir.exists():
        alt = (Path(__file__).resolve().parent / 'tobacco3.0').resolve()
        if alt.exists(): tob_dir = alt
        else:
            raise FileNotFoundError(f"tobacco3.0 not found at {args.tobacco_dir}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    # Ensure template in templates/
    src_template = tob_dir / 'template_database' / f'{args.net}.cif'
    if not src_template.exists(): raise FileNotFoundError(f"Template for net '{args.net}' not found at {src_template}")
    dst_templates = tob_dir / 'templates'
    dst_templates.mkdir(exist_ok = True)
    with open(src_template, 'r') as r, open(dst_templates / f'{args.net}.cif', 'w') as w: w.write(r.read())

    nodes_dir = tob_dir / 'nodes'
    nodes_dir.mkdir(exist_ok = True)
    node_input_cif = nodes_dir / 'input_node.cif'
    if node_input_cif.exists():
        try: node_input_cif.unlink()
        except Exception: pass

    if args.node_cif:
        node_src = Path(args.node_cif)
        if not node_src.exists(): raise FileNotFoundError(f"node-cif not found: {node_src}")
        shutil.copyfile(str(node_src), str(node_input_cif))
        print("  -> using provided node CIF")

    edges_dir = tob_dir / 'edges'
    edges_dir.mkdir(exist_ok = True)
    output_cifs = tob_dir / 'output_cifs'
    output_cifs.mkdir(exist_ok = True)

    input = pd.read_csv(args.input)
    node = input['scaffold'][0]
    assert node is not None, "No organic node SMILES found"
    print(f"[build] Organic node: {node}")
    if not args.node_cif:
        try:
            node_atoms = generate_node_atoms_from_smiles(node)
            write_cif_from_atoms(node_atoms, str(node_input_cif))
            print("  -> organic node prepared")
        except Exception as e:
            print(f"  -> fail (node prepare failed): {e}")
            return 1
    edge_list = input['arm'].tolist()
    if not edge_list:
        print(f"No SMILES found in {args.input}")
        return 0
    
    for idx, smi in enumerate(edge_list, start = 1):
        print(f"[build {idx}/{len(edge_list)}] {smi}")
        for old_edge in edges_dir.glob('edge_*.cif'):
            try: old_edge.unlink()
            except Exception: pass
    
        input_edge_path = edges_dir / 'input_edge.cif'
        if input_edge_path.exists():
            try: input_edge_path.unlink()
            except Exception: pass
    
        try:
            linker_atoms = generate_edge_atoms_from_smiles(smi)
            named_edge = edges_dir / ( _sanitize_for_filename(f"edge_{idx:03d}__{smi}") + ".cif" )
            write_cif_from_atoms(linker_atoms, str(named_edge))
            shutil.copyfile(str(named_edge), str(edges_dir / 'input_edge.cif'))
            print("  -> edge prepared")
        except Exception as e:
            print(f"  -> skip (prepare failed): {e}")
            continue

        try: 
            run_tobacco(args.tobacco_dir, python_exe = args.tobacco_py)
        except Exception as e:
            print(f"  -> fail (ToBaCCo): {e}")
            continue

        print("  -> cleaning up edge files...")
        for edge_file in edges_dir.glob('*.cif'):
            try: edge_file.unlink()
            except Exception: pass

        produced = sorted(output_cifs.glob('*.cif'))
        if not produced: print("  -> warn: no output_cifs produced")
        else: print(f"  -> success: {len(produced)} structures generated")
        
        for cif in produced:
            name = _sanitize_for_filename(f"{args.net}__{idx:03d}__{smi}") + ".cif"
            dest = out_dir / name
            try:
                os.replace(str(cif), str(dest))
                print(f"  -> saved: {dest.name}")
            except Exception:
                shutil.copyfile(str(cif), str(dest))
                try:
                    cif.unlink()
                except Exception:
                    pass
                
    for f in output_cifs.glob('*.cif'):
        try: f.unlink()
        except Exception: pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
