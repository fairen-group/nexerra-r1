# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# smiles -> mol
# Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

from rdkit import Chem
from rdkit.Chem import AllChem
import os
import sys

def smi2mol(smi: str, output_dir: str):
    '''Convert SMILES to MOL file with 2D coordinates'''
    mol = Chem.MolFromSmiles(smi)
    if mol is None: raise ValueError(f"Invalid SMILES: {smi}")
    AllChem.Compute2DCoords(mol)
    output_path = os.path.join(output_dir, Chem.MolToSmiles(mol) + ".mol")
    writer = Chem.SDWriter(output_path)
    writer.write(mol)
    writer.close()

if __name__ == "__main__":
    input = sys.argv[1]
    assert os.path.exists(input), f"Input file {input} does not exist"
    output_dir = sys.argv[2]
    assert os.path.isdir(output_dir), f"Output directory {output_dir} does not exist"

    smiles = []
    with open(input, 'r') as f: 
        for line in f: smiles.append(line.strip())
    
    for smi in smiles:
        try: smi2mol(smi, output_dir)
        except Exception as e: print(f"Error processing {smi}: {e}", file=sys.stderr)