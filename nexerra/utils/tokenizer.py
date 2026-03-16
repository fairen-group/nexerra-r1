# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Tokenizer for SELFIES
# Note: Used during inference and testing, for preparing dataset, use preprocess.py
# Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import pickle
import argparse

from rdkit import Chem
import selfies as sf

# -----------------------
# Tokenizer class 
# -----------------------

class Tokenizer:
    def __init__(self, tok2idx: dict = None, idx2tok: dict = None, max_len: int = None):
        self.tok2idx = tok2idx
        self.idx2tok = idx2tok
        self.max_len = max_len

    def encode_single(self, smi):
        '''Encodes a single SMILES string into a padded tensor of token IDs'''
        try: selfies = sf.encoder(smi)
        except Exception as e: raise ValueError(f"Failed to encode SMILES to SELFIES: {smi}. Error: {e}")

        tokens = ["[SOS]"] + list(sf.split_selfies(selfies)) + ["[EOS]"]
        token_ids = [self.tok2idx.get(tok, self.tok2idx["[UNK]"]) for tok in tokens]
        token_ids += [self.tok2idx["[PAD]"]] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids[:self.max_len], dtype = torch.long)

    def decode_single(self, token_ids):
        '''Decodes a tensor of token IDs back to a SMILES string'''
        if isinstance(token_ids, torch.Tensor): token_ids = token_ids.tolist()
        tokens = [self.idx2tok[int(i)] for i in token_ids if self.idx2tok[int(i)] not in ["[SOS]", "[EOS]", "[PAD]", "[UNK]"]]
        selfies_str = "".join(tokens)
        try: return sf.decoder(selfies_str)
        except Exception as e: raise ValueError(f"Failed to decode SELFIES: {selfies_str}. Error: {e}")

def test_tokenizer(input_smiles: str, tok2idx: dict, idx2tok: dict, max_len: int):
    '''Unittest on the tokenizer class'''
    tokenizer = Tokenizer(tok2idx, idx2tok, max_len)
    encoded = tokenizer.encode_single(input_smiles)
    decoded = tokenizer.decode_single(encoded)

    print(f"Input SMILES: {input_smiles}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    inp_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(input_smiles), isomericSmiles = True, canonical = True)
    dec_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(decoded), isomericSmiles = True, canonical = True)
    assert dec_canonical == inp_canonical, "Decoded SMILES does not match the original input."
    print("Tokenizer test passed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenizer (test)')
    parser.add_argument('--files', type = str, required = True, help = 'Path to tokenizer file')
    args = parser.parse_args()

    with open(args.files, 'rb') as f: dataset = pickle.load(f)
    
    tok2id = dataset['tok2id']
    id2tok = dataset['id2tok']
    max_len = dataset['max_len']

    input_smiles = "O=C(O)c7ccc(c6cc(c1ccc(C(=O)O)cc1)cc(c5ccc(c4cc(c2ccc(C(=O)O)cc2)cc(c3ccc(C(=O)O)cc3)c4)cc5)c6)cc7"
    test_tokenizer(input_smiles, tok2id, id2tok, max_len)

