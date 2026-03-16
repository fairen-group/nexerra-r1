# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Complete pre-processing of raw training data.
# SMILES -> SELFIES -> Tokenization (with padding and special tokens).
# [Lr] -> connection points to the vertex (either organic or metallic) - preserved.
# Dataset enumeration function included - increase the number of molecules for training. 
# Author: Dhruv Menon (dm958[at]cam.ac.uk)
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

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Standard libraries ---
import pandas as pd
import pickle
import json
from tqdm import tqdm
import numpy as np
if not hasattr(np, 'bool'): np.bool = np.bool_ # For compatibility with the SCScorer library. Workaround.
import random; random.seed(42) # for reproducibility
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(handler)

# --- RDKit and SELFIES ---
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit import RDLogger
# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')
import selfies as sf

# --- Pytorch ---
import torch

import pyfiglet
def display_banner():
    banner = pyfiglet.figlet_format("Nexerra", font="slant")
    print(banner)

# --- Dataset class ---
class SELFIESDataset:
    def __init__(self, smiles_list: list, tok2idx: dict = None, max_len: int = None):
        self.raw_smiles = smiles_list
        self.canonical_smiles = []

        # --- Filter out duplicate SMILES ---
        logger.info(f"Canonicalizing SMILES strings.")
        for smi in tqdm(self.raw_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            canonical_smi = Chem.MolToSmiles(mol, canonical = True)
            self.canonical_smiles.append(canonical_smi)
        self.canonical_smiles = list(set(self.canonical_smiles))  # Remove duplicates
        logger.info(f"Number of total SMILES: {len(self.raw_smiles)}")
        logger.info(f"Number of unique SMILES: {len(self.canonical_smiles)}")
        
        # --- SMILES -> SELFIES -> Tokenize ---
        logger.info(f"Converting SMILES to SELFIES.")
        
        self.selfies_list = []
        valid_canonical_smiles = []
        failed_selfies = 0
        for smi in self.canonical_smiles:
            if Chem.MolFromSmiles(smi) is None: continue
            try:
                self.selfies_list.append(sf.encoder(smi))
                valid_canonical_smiles.append(smi)
            except Exception as e:
                failed_selfies += 1
                logger.warning("Failed to encode SMILES to SELFIES: %s | %s", smi, e)
                continue
        self.canonical_smiles = valid_canonical_smiles
        logger.info("Number of valid SELFIES: %d", len(self.selfies_list))
        logger.info("Number of SELFIES encoding failures: %d", failed_selfies)
        logger.info(f"Tokenizing SELFIES.")
        self.tokenized = [list(sf.split_selfies(smi)) for smi in self.selfies_list]
        
        # --- Build vocabulary ---
        tokens = set(tok for toks in self.tokenized for tok in toks) # Unique tokens
        self.sos = '[SOS]' # Start of sequence token
        self.eos = '[EOS]' # End of sequence token
        self.padding = '[PAD]' # Padding token
        self.unk = '[UNK]' # Unknown token
        self.special_tokens = [self.sos, self.eos, self.padding, self.unk]
        self.vocab = self.special_tokens + sorted(tokens) if tok2idx is None else list(tok2idx.keys())
        self.tok2idx = {tok : idx for idx, tok in enumerate(self.vocab)} # Token to index mapping
        self.idx2tok = {idx : tok for tok, idx in self.tok2idx.items()} # Index to token mapping

        if max_len is None: # For padding shorter sequences
            token_lengths = [len(list(t)) for t in self.tokenized]
            self.max_len = max(token_lengths) + 2  # +2 for [SOS] and [EOS] 
        else:
            self.max_len = max_len
        
    def encode_selfies(self, tokens):
        '''Encodes a list of SELFIES tokens into a padded tensor of token IDs'''
        tokens = [self.sos] + list(tokens) + [self.eos]
        token_ids = [self.tok2idx.get(tok, self.tok2idx[self.unk]) for tok in tokens]
        token_ids += [self.tok2idx[self.padding]] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids[: self.max_len], dtype = torch.long)

    def decode_selfies(self, token_ids):
        '''Decodes a tensor of token IDs back to a SELFIES string'''
        tokens = [self.idx2tok[int(i)] for i in token_ids if self.idx2tok[int(i)] not in [self.sos, self.eos, self.padding, self.unk]]
        selfies_str = "".join(tokens)
        try: return sf.decoder(selfies_str)
        except: return None

    def __len__(self):
        return len(self.tokenized)
    
    def __getitem__(self, idx):
        token_ids = self.encode_selfies(self.tokenized[idx])
        return {"input" : token_ids}
        
    def get_vocab(self): return self.tok2idx, self.idx2tok
    
    def get_vocab_size(self): return len(self.vocab)
    
    def special_token_ids(self):
        sos_index = self.tok2idx[self.sos]
        eos_index = self.tok2idx[self.eos]
        padding_index = self.tok2idx[self.padding]
        unk_index = self.tok2idx[self.unk]
        return sos_index, eos_index, padding_index, unk_index

# --- Compute properties & normalize (independent of tokenization) ---
def compute_properties(smi):
        clean_smi = smi.replace('[Lr]', '[*]') # Replace [Lr] with dummy atom for compatibility
        mol = Chem.MolFromSmiles(clean_smi)
        assert mol is not None, "Bad SMILES: " + smi + " --> fix!"
        logP = Crippen.MolLogP(mol)
        return float(logP)

def normalize(raw_prop: float, stats: dict):
    return (raw_prop - stats["mean"]) / stats["std"]

def denormalize(norm_prop: float, stats: dict):
    return norm_prop * stats["std"] + stats["mean"]

# ----------------------------------------------------
# UNIT TESTS (for internal testing)
# ----------------------------------------------------

def run_unittest(src: str) -> None:
    '''
    unit test for preprocess/trainer compatibility
    ---
    i. tokenizer sanity
    ii. serialized artifact schema
    iii. trainer/dataLoader compatibility smoke test'''
    import tempfile
    from nexerra.model.DataLoader import TrainDataLoader

    logger.info("Running preprocess unit tests using %s", src)
    data = pd.read_csv(src)
    if "smiles" not in data.columns:
        raise AssertionError("Input CSV must contain a 'smiles' column.")

    smiles = data["smiles"].dropna().astype(str).tolist()
    if len(smiles) == 0:
        raise AssertionError("Input CSV contains no SMILES.")

    dataset = SELFIESDataset(smiles)
    if len(dataset) == 0:
        raise AssertionError("Dataset is empty after preprocessing.")

    tok2id, id2tok = dataset.get_vocab()
    vocab_size = dataset.get_vocab_size()
    max_len = dataset.max_len
    sos_index, eos_index, padding_index, unk_index = dataset.special_token_ids()

    # -------------------------------------------------
    # i. round-trip tokenizer test
    # -------------------------------------------------
    logger.info("Test 1/3 - round-trip tokenizer [...]")

    checked = 0
    for smi in dataset.canonical_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue

        canonical_in = Chem.MolToSmiles(mol, canonical = True)
        selfies_str = sf.encoder(canonical_in)
        token_ids = dataset.encode_selfies(list(sf.split_selfies(selfies_str)))
        decoded = dataset.decode_selfies(token_ids.tolist())

        if decoded is None:
            raise AssertionError(f"Decoded SMILES is None for input: {smi}")

        mol_dec = Chem.MolFromSmiles(decoded)
        if mol_dec is None:
            raise AssertionError(f"Decoded SMILES is invalid for input: {smi} -> {decoded}")

        canonical_out = Chem.MolToSmiles(mol_dec, canonical=True)
        if canonical_in != canonical_out:
            raise AssertionError(
                f"Round-trip mismatch: input={canonical_in} output={canonical_out}"
            )

        checked += 1
        if checked >= min(10, len(dataset.canonical_smiles)):
            break

    if checked == 0:
        raise AssertionError("No valid molecules available for round-trip test.")

    logger.info("Round-trip tokenizer test passed on %d samples", checked)

    # -------------------------------------------------
    # ii. serialized artifact schema test
    # courtesy: some help from codex...
    # -------------------------------------------------
    logger.info("Test 2/3: serialized artifact schema")

    encoded_inputs = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if "input" not in sample:
            raise AssertionError(f"Dataset item {i} missing 'input'")
        encoded_inputs.append(sample["input"])

    encoded_inputs = torch.stack(encoded_inputs)

    tokenized_dataset = {
        "encoded_dataset": encoded_inputs,
        "vocab_size": vocab_size,
        "max_len": max_len,
        "tok2id": tok2id,
        "id2tok": id2tok,
        "sos_index": sos_index,
        "eos_index": eos_index,
        "padding_index": padding_index,
        "unk_index": unk_index,
    }

    required_keys = {
        "encoded_dataset",
        "vocab_size",
        "max_len",
        "tok2id",
        "id2tok",
        "sos_index",
        "eos_index",
        "padding_index",
        "unk_index",
    }

    if set(tokenized_dataset.keys()) != required_keys:
        missing = required_keys - set(tokenized_dataset.keys())
        extra = set(tokenized_dataset.keys()) - required_keys
        raise AssertionError(f"Schema mismatch. Missing={missing}, Extra={extra}")

    if tokenized_dataset["encoded_dataset"].dtype != torch.long:
        raise AssertionError("encoded_dataset must be torch.long")

    if tokenized_dataset["encoded_dataset"].ndim != 2:
        raise AssertionError("encoded_dataset must be rank-2 [N, max_len]")

    if tokenized_dataset["encoded_dataset"].shape[1] != tokenized_dataset["max_len"]:
        raise AssertionError("encoded_dataset width must equal max_len")

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = os.path.join(tmpdir, "tokenized_dataset.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(tokenized_dataset, f)

        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        if set(loaded.keys()) != required_keys:
            missing = required_keys - set(loaded.keys())
            extra = set(loaded.keys()) - required_keys
            raise AssertionError(f"Loaded schema mismatch. Missing={missing}, Extra={extra}")

        if loaded["encoded_dataset"].shape[1] != loaded["max_len"]:
            raise AssertionError("Serialized encoded_dataset width must equal serialized max_len")

        logger.info("Serialized artifact schema test passed")

        # -------------------------------------------------
        # iii. Trainer/DataLoader compatibility smoke test
        # -------------------------------------------------
        logger.info("Test 3/3: trainer compatibility smoke test")

        loader = TrainDataLoader(pkl_path)
        if len(loader) == 0:
            raise AssertionError("TrainDataLoader loaded an empty dataset")

        sample = loader[0]
        if "input" not in sample or "target" not in sample:
            raise AssertionError("TrainDataLoader sample must contain 'input' and 'target'")

        if sample["input"].shape != sample["target"].shape:
            raise AssertionError("input and target shapes must match")

        if sample["input"].dtype != torch.long or sample["target"].dtype != torch.long:
            raise AssertionError("input and target tensors must be torch.long")

        if sample["input"].numel() != loaded["max_len"]:
            raise AssertionError("Sample tensor length must match max_len")

        logger.info("Trainer compatibility smoke test passed")

    logger.info("All preprocess unit tests passed successfully.")

        
# --- Main ---
if __name__ == '__main__':
    display_banner()
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--mode', type=str, required=True, choices = ['gen', 'prop', 'unittest'], help='gen for tokenizing dataset; prop for ONLY caching properties, unittest for unit test on full tokenization loop')
    parser.add_argument('--src', type=str, help='Path to the training dataset')
    parser.add_argument('--dst', type=str, help='Path to save the tokenized dataset')
    args = parser.parse_args()

    if args.mode == 'gen':
        logger.info(f"Tokenizing dataset from {args.src}")
        data = pd.read_csv(args.src)
        smiles = data['smiles'].tolist()

        dataset = SELFIESDataset(smiles)
        if len(dataset) == 0: raise ValueError("The dataset is empty after canonicalization/SELFIES encoding.")
        tok2id, id2tok = dataset.get_vocab()
        vocab_size = dataset.get_vocab_size()
        logger.info(f"Vocabulary size: {vocab_size}")
        max_len = dataset.max_len
        logger.info(f"Max sequence length: {max_len}")

        # Pre-tokenize & cache
        encoded_inputs = []
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            encoded_inputs.append(sample["input"])
        
        encoded_inputs = torch.stack(encoded_inputs)
        sos_index, eos_index, padding_index, unk_index = dataset.special_token_ids()

        tokenized_dataset = {
            'encoded_dataset': encoded_inputs,
            'vocab_size': vocab_size,
            'max_len': max_len,
            'tok2id': tok2id,
            'id2tok': id2tok,
            'sos_index': sos_index,
            'eos_index': eos_index,
            'padding_index': padding_index,
            'unk_index': unk_index
        }

        with open(os.path.join(args.dst, 'tokenized_dataset.pkl'), 'wb') as f:
            pickle.dump((tokenized_dataset), f)
        
        with open(os.path.join(args.dst, 'id2tok.json'), "w") as f:
            json.dump(id2tok, f)
        
        with open(os.path.join(args.dst,'tok2id.json'), "w") as f:
            json.dump(tok2id, f)

        torch.save(encoded_inputs, os.path.join(args.dst,'cached_inputs.pt'))
        torch.save(tok2id, os.path.join(args.dst, 'token2idx.pt'))
        torch.save(id2tok, os.path.join(args.dst, 'idx2token.pt'))
    
    if args.mode == 'prop':
        logger.info(f"Calculating logP for dataset from {args.src}")
        data = pd.read_csv(args.src)
        smiles = data['smiles'].tolist()
        properties = []
        for smi in tqdm(smiles): properties.append(compute_properties(smi))
        assert len(properties) == len(smiles), "Mismatch in number of properties and SMILES"
        mu, sigma = float(np.mean(properties)), float(np.std(properties))
        stat_path = args.dst + "_stats.json"
        with open(stat_path, "w") as f: json.dump({"mean": mu, "std": max(1e-8, sigma)}, f, indent = 2)
        logger.info(f"Properties stats (for normalization) saved to:", {stat_path})
        normalized_props = [normalize(p, {"mean": mu, "std": sigma}) for p in properties]
        props = torch.tensor(normalized_props, dtype = torch.float32)
        torch.save(props, os.path.join(args.dst,'cached_logP.pt'))
        logger.info(f"Properties saved to {os.path.join(args.dst)}")
    
    if args.mode == 'unittest':
        logger.info("running unit tests")
        run_unittest(src = args.src)
