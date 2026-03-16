# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  Add hydrogens to cifs 
#  Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import argparse
import sys
from openbabel import pybel

def add_hydrogens_to_cif(input_cif: str, output_cif: str) -> None:
    mols = list(pybel.readfile("cif", input_cif))
    if not mols: raise ValueError(f"No structures found in CIF file: {input_cif}")

    for mol in mols: mol.addh()

    with open(output_cif, "w") as fout:
        first = True
        for mol in mols:
            cif_block = mol.write("cif")
            if not first and not cif_block.startswith("data_"):
                fout.write("\n")
            fout.write(cif_block)
            first = False
    print(f"Hydrogens added. Wrote: {output_cif}")


def main():
    parser = argparse.ArgumentParser(description = "Add hydrogens to a MOF CIF file using Open Babel")
    parser.add_argument("input_cif", help = "Input CIF file (without hydrogens)")
    parser.add_argument("output_cif", help="Output CIF file (with hydrogens)")
    args = parser.parse_args()
    try: add_hydrogens_to_cif(args.input_cif, args.output_cif)
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n"); sys.exit(1)

if __name__ == "__main__": main()
