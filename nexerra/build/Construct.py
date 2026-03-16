# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  Batch assembly of MOFs using generated linkers & pre-defined nets via ToBaCCo3.0
#  Example usage: 
#  python Construct.py --input [path_to_smiles.txt] --tobacco-dir [path_to_tobacco3.0] 
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
# Most importantly, this is likely to work with single node single edge cases; for cases whre there is an organic node;
# look at ScaffConstruct.py. Note that for this you need to change tobacco settings.
# Credits for the TobaCCo3.0 code: "Cryst. Growth Des. 2017, 17, 11, 5801–5810"
# Place Tobacco3.0 in this directory, name the folder 'tobacco3.0'
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import re
import math
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Set

# --- RDKit ---
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import pyfiglet

# Silence noisy RDKit warnings from dummy atoms during UFF attempts.
RDLogger.DisableLog("rdApp.warning")

def display_banner() -> None: banner = pyfiglet.figlet_format("Nexerra", font = "slant"); print(banner)

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
    def _clean_token(tok: str) -> str: return tok.strip().strip('"').strip("'")

    def _pick_smiles_from_line(line: str) -> str:
        if "," in line or "\t" in line:
            fields = [_clean_token(x) for x in re.split(r"[\t,]", line)]
            anchor_fields = [f for f in fields if "[Lr]" in f]
            if anchor_fields: return anchor_fields[0]
            # Fallback to first non-empty field to preserve previous behavior.
            for f in fields:
                if f: return f
            return ""
        return _clean_token(line)

    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            t = raw.strip()
            if not t or t.startswith("#"): continue
            t = _pick_smiles_from_line(t)
            if not t: continue
            # Skip common header rows.
            if line_num == 1 and t.lower() in {"smiles", "linker", "linkers"}: continue
            lines.append(t)
    return lines

# --- Write CIFs for ToBaCCo edges ---
def write_cif_from_atoms(atoms: List, cif_path: str, box_size: float = 50.0) -> None:
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
        x_count = 0
        other_counts: dict = {}

        # --- For ToBaCCo, connectors are labeled 'X' ---
        def is_connector_symbol(s: str) -> bool:
            us = s.upper()
            return s == "[Lr]" or us == "X" or us == "XE"

        for idx, (sym, _, _, _) in enumerate(atoms, start = 1):
            x, y, z = frac[idx - 1]
            if is_connector_symbol(sym):
                x_count += 1
                label = f"X{x_count}"
                labels.append(label)
                w.write(f"{label} C {x:.6f} {y:.6f} {z:.6f}\n")
            else:
                esym = sym
                other_counts[esym] = other_counts.get(esym, 0) + 1
                label = f"{esym}{other_counts[esym]}"
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


def generate_edge_atoms_from_smiles(smiles: str, num_confs: int = 20, seed: int = 42) -> List[Tuple[str, float, float, float]]:
    '''Generate a 3D conformer from linker SMILES for ToBaCCo3.0 edges
    ---
      - Replace [Lr] with RDKit dummy '*'.
      - Embed and optimize a conformer.
      - For each '*': if in ring, keep and label 'X'; else move connector to its single neighbor (label neighbor 'X') and drop the '*'.
    '''
    smi = smiles.replace("[Lr]", "*")
    mol = Chem.MolFromSmiles(smi)
    if mol is None: raise ValueError("Failed to parse SMILES (after [Lr]->* replacement).")
    
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, num_confs), params=params))
    if not conf_ids:
        raise ValueError("Failed to embed a 3D conformer for this SMILES.")
    try: AllChem.UFFOptimizeMoleculeConfs(mol, confIds=conf_ids)
    except Exception: pass

    anchor_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    body_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 0]
    if len(anchor_idxs) < 1 or not body_idxs: raise ValueError("SMILES must contain at least one [Lr] anchor and other atoms")

    conf = mol.GetConformer(conf_ids[0])
    connector_idxs: List[int] = []
    drop_anchor: Set[int] = set()
    anchor_connector_idxs: Set[int] = set()
    for ai in anchor_idxs:
        a = mol.GetAtomWithIdx(ai)
        if a.IsInRing():
            connector_idxs.append(ai)
            anchor_connector_idxs.add(ai)
        else:
            nbrs = [b.GetOtherAtomIdx(ai) for b in a.GetBonds()]
            if len(nbrs) != 1: raise ValueError(f"Non-ring anchor at idx {ai} has {len(nbrs)} neighbors; expected exactly 1.")
            ni = nbrs[0]
            connector_idxs.append(ni)
            drop_anchor.add(ai)

    atoms: List[Tuple[str, float, float, float]] = []
    connector_set = set(connector_idxs)
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        if idx in drop_anchor: continue
        
        # Remove H only from anchor-style X sites (retained dummy anchors).
        # Keep H on neighbor-mapped X sites to preserve valence in unsaturated motifs.
        
        if a.GetAtomicNum() == 1:
            nbrs = [b.GetOtherAtomIdx(idx) for b in a.GetBonds()]
            if any(n in anchor_connector_idxs for n in nbrs):
                continue
        pos = conf.GetAtomPosition(idx)
        sym = 'X' if idx in connector_idxs else a.GetSymbol()
        atoms.append((sym, float(pos.x), float(pos.y), float(pos.z)))
    return atoms


def run_tobacco(tob_dir: Path, python_exe: Optional[str] = None) -> None:
    script = 'tobacco.py'
    if not (tob_dir / script).exists():
        raise FileNotFoundError(f"Expected {script} in {tob_dir}, but it was not found")

    # Interpret --tobacco-py as python interpreter path; correct common misuse
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


def _clear_cifs(directory: Path) -> None:
    for f in directory.glob("*.cif"):
        try: f.unlink()
        except Exception: pass


def _collect_produced_output(output_cifs: Path) -> Optional[Path]:
    produced = sorted(output_cifs.glob('*.cif'))
    if not produced:
        print("  -> warn: no output_cifs produced"); return None
    if len(produced) > 1:
        print(f"  -> warn: expected 1 output CIF, found {len(produced)}; using first: {produced[0].name}")
    return produced[0]


def main(argv: Optional[List[str]] = None) -> int:
    display_banner()
    ap = argparse.ArgumentParser(description="Simple SMILES --> ToBaCCo --> output .cif pipeline")
    ap.add_argument('--input', required=True, help='Path to .txt/.csv with SMILES containing [Lr] anchors')
    ap.add_argument('--tobacco-dir', required=True, help='Path to tobacco3.0 directory')
    ap.add_argument('--net', required = True, help='Net name (must exist in template_database/<net>.cif)')
    ap.add_argument('--output-dir', required = True, help='Directory to place produced .cif files')
    ap.add_argument('--tobacco-py', default = None, help='Python executable to run tobacco.py (defaults to current env)')
    ap.add_argument('--node-cif', default = None, help='Optional path to node CIF to copy once to nodes/input_node.cif')
    args = ap.parse_args(argv)

    tob_dir = Path(args.tobacco_dir).resolve()
    if not tob_dir.exists():
        alt = (Path(__file__).resolve().parent / 'tobacco3.0').resolve()
        if alt.exists(): tob_dir = alt
        else: raise FileNotFoundError(f"tobacco3.0 not found at {args.tobacco_dir}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    # Ensure template in templates/
    src_template = tob_dir / 'template_database' / f'{args.net}.cif'
    if not src_template.exists(): raise FileNotFoundError(f"Template for net '{args.net}' not found at {src_template}")
    dst_templates = tob_dir / 'templates'
    dst_templates.mkdir(exist_ok=True)
    _clear_cifs(dst_templates)
    with open(src_template, 'r') as r, open(dst_templates / f'{args.net}.cif', 'w') as w: w.write(r.read())

    if args.node_cif:
        node_src = Path(args.node_cif)
        if not node_src.exists(): raise FileNotFoundError(f"node-cif not found: {node_src}")
        nodes_dir = tob_dir / 'nodes'
        nodes_dir.mkdir(exist_ok = True)
        shutil.copyfile(str(node_src), str(nodes_dir / 'input_node.cif'))

    edges_dir = tob_dir / 'edges'
    edges_dir.mkdir(exist_ok = True)
    output_cifs = tob_dir / 'output_cifs'
    output_cifs.mkdir(exist_ok = True)

    smiles_list = _read_smiles_lines(args.input)
    if not smiles_list:
        print(f"No SMILES found in {args.input}")
        return 0

    for idx, smi in enumerate(smiles_list, start=1):
        print(f"[build {idx}/{len(smiles_list)}] {smi}")
        _clear_cifs(edges_dir)
        _clear_cifs(output_cifs)

        try:
            linker_atoms = generate_edge_atoms_from_smiles(smi)
            write_cif_from_atoms(linker_atoms, str(edges_dir / 'input_edge.cif'))
        except Exception as e:
            print(f"-> skip (prepare failed): {e}")
            continue

        try:
            run_tobacco(tob_dir, python_exe = args.tobacco_py)
        except Exception as e:
            print(f"-> fail (ToBaCCo): {e}")
            continue

        # Move outputs
        cif = _collect_produced_output(output_cifs)
        if cif is not None:
            name = _sanitize_for_filename(f"{args.net}__{idx:03d}__{smi}") + ".cif"
            dest = out_dir / name
            try:
                os.replace(str(cif), str(dest))
            except Exception:
                shutil.copyfile(str(cif), str(dest))
                try: cif.unlink()
                except Exception: pass
        # Clean for next run
        for f in output_cifs.glob('*.cif'):
            try: f.unlink()
            except Exception: pass

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
