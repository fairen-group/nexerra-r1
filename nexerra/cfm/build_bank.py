# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Build the latent bank for training the OT-CFM model
# Example usage:
#   python build_bank.py --mode build --data [path_to_dataset.csv] \
#           --mparams [path_to_mparams.pkl] --batch_size [batch_size] --savepath [path_to_save_path]
# Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Standard library imports ---
import logging
import argparse
import numpy as np
import pickle
import tempfile
from tqdm import tqdm
import random
random.seed(42)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(handler)

# --- PyTorch imports ---
import torch
torch.manual_seed(42)

# --- Local imports ---
from nexerra.model.HTVAE import VAEModel
from nexerra.utils.tokenizer import Tokenizer
from nexerra.inference.Reward import RewardFunction 

import pyfiglet
def display_banner(): banner = pyfiglet.figlet_format("Nexerra", font = "slant"); print(banner)

# --- Construct latent bank {Z, Y} for training the OT-CFM model ---
@torch.no_grad() # Freeze the model parameters
def build_bank(model: VAEModel, 
               tokenizer: Tokenizer,
               rf: RewardFunction, 
               dataset: list,
               batch_size: int = 128,
               savepath: str = 'latent_bank.pt',  
               device: str = 'cpu'):
    '''Build the latent bank {Z, Y} for training the OT-CFM model'''
    Z, Y = [], []
    reward_cache = []
    model.eval()
    skipped = 0 # <-- count skipped molecules due to reward function errors
    for i in tqdm(range(0, len(dataset), batch_size), desc = "Encoding dataset into latent space"):
        batch = dataset[i: i + batch_size]
        # --- Calculate the rewards ---
        valid_rewards = []
        valid_smiles = []
        for smi in batch:
            try:
                reward = rf.anc_to_anc_length(smi)
                if not np.isnan(reward) and not np.isinf(reward):
                    valid_rewards.append(reward)
                    valid_smiles.append(smi)
                    reward_cache.append(reward)
                else:
                    skipped += 1
                    logger.debug(f"Skipped SMILES due to invalid reward: {smi}")
            except Exception as e:
                skipped += 1
                logger.debug(f"Skipped SMILES due to error in reward calculation: {smi} | Error: {e}")
        
        # --- Skip batch if NO valid molecules ---
        if not valid_smiles: continue

        # --- Tokenize the valid SMILES ---
        tokens = [tokenizer.encode_single(smi) for smi in valid_smiles]
        X = torch.stack(tokens).to(device)
        # --- Calculate the padding mask ---
        src_key_padding_mask = (X == model.pad_index)
        # --- Forward pass ---
        _, mu, _ = model.encode(src = X, src_mask = None, src_key_padding_mask = src_key_padding_mask)
        # --- Append mu (instead of z) to the bank to reduce noise ---
        Z.append(mu.detach().cpu().float())
        # --- Append rewards ---
        Y.append(torch.tensor(valid_rewards, dtype = torch.float32))
    
    if not Z: raise ValueError("No valid molecules found in the dataset to build the latent bank.")
    
    Z = torch.cat(Z, dim = 0)
    Y = torch.cat(Y, dim = 0)
    # --- Save the latent bank ---
    torch.save({"Z": Z, "Y": Y}, savepath)
    logger.info(f"Latent bank saved at: {savepath}")
    # --- Save the rewards ---
    reward_path = savepath.replace('latent_bank.pt', 'rewards.pt')
    torch.save(torch.tensor(reward_cache, dtype=torch.float32), reward_path)
    logger.info(f"Rewards saved at: {reward_path}")

    logger.info(f"Total valid molecules: {len(reward_cache)}, Skipped: {skipped}")
    if reward_cache:
        logger.info(f"Reward range: [{min(reward_cache):.4f}, {max(reward_cache):.4f}]")
        logger.info(f"Mean reward: {np.mean(reward_cache):.4f}, Std: {np.std(reward_cache):.4f}")

    return Z.shape, Y.shape


@torch.no_grad()
def run_unittest(model: VAEModel,
                 tokenizer: Tokenizer,
                 rf: RewardFunction,
                 dataset: list[str],
                 device: str = 'cpu'):
    '''test for latent bank construction on a dummy dataset'''
    if not dataset: raise ValueError("Dataset is empty; cannot run build_bank unittest.")

    sample_size = min(32, len(dataset))
    sample_dataset = dataset[:sample_size]
    logger.info("Running build_bank unittest on %d SMILES.", len(sample_dataset))

    with tempfile.TemporaryDirectory() as tmpdir:
        savepath = os.path.join(tmpdir, 'latent_bank.pt')
        z_shape, y_shape = build_bank(
            model = model,
            tokenizer = tokenizer,
            rf = rf,
            dataset = sample_dataset,
            batch_size = min(8, len(sample_dataset)),
            savepath = savepath,
            device = device,
        )

        if not os.path.exists(savepath): raise AssertionError("Latent bank file was not created")

        bank = torch.load(savepath, map_location = 'cpu')
        if "Z" not in bank or "Y" not in bank:
            raise AssertionError("Latent bank is missing required keys: Z and Y.")
        if bank["Z"].ndim != 2:
            raise AssertionError("Z must be rank-2 [N, latent_dim].")
        if bank["Y"].ndim != 1:
            raise AssertionError("Y must be rank-1 [N].")
        if bank["Z"].shape[0] != bank["Y"].shape[0]:
            raise AssertionError("Z and Y must contain the same number of rows.")
        if tuple(bank["Z"].shape) != tuple(z_shape):
            raise AssertionError("Saved Z shape does not match returned shape.")
        if tuple(bank["Y"].shape) != tuple(y_shape):
            raise AssertionError("Saved Y shape does not match returned shape.")

        reward_path = savepath.replace('latent_bank.pt', 'rewards.pt')
        if not os.path.exists(reward_path):
            raise AssertionError("Reward cache file was not created.")

    logger.info("build_bank unittest passed successfully.")

# --- Main function ---
if __name__ == "__main__":
    display_banner()
    ap = argparse.ArgumentParser(description = "Build the latent bank for training the OT-CFM model")
    ap.add_argument('--mode', type = str, default = 'build', choices = ['build', 'test'], help = 'build: construct and save the latent bank; test: run a smoke test on the provided dataset')
    ap.add_argument('--data', type = str, required = True, help = 'Path to the training dataset')
    ap.add_argument('--mparams', type = str, required = False, help = 'Path to model params (if changed from default)')
    ap.add_argument('--batch_size', type = int, default = 128, help = 'Batch size for encoding')
    ap.add_argument('--savepath', type = str, default = 'latent_bank.pt', help = 'Path to save the latent bank')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --- hard-coded paths (don't change unless the ckpts are elsewhere) ---
    ckpt_path = '../../artifacts/ckpt/vae/no_prop_vae_epoch_120.pt'
    if not args.mparams: mparams = '../../data/processed/tokenized_dataset.pkl'
    else: mparams = args.mparams
    with open(mparams, 'rb') as f: dataset = pickle.load(f)
    tok2id = dataset['tok2id']
    id2tok = dataset['id2tok']
    max_len = dataset['max_len']
    vocab_size = dataset['vocab_size']
    sos_index = dataset['sos_index']
    eos_index = dataset['eos_index']
    padding_index = dataset['padding_index']
    unk_index = dataset['unk_index']
    
    # ---- Model initialization ----
    model = VAEModel(
        device = device,
        shape_flag = 0,
        vocab_size = vocab_size,
        sos_index = sos_index,
        eos_index = eos_index,
        pad_index = padding_index,
        unk_index = unk_index,
        d_model = 512,
        latent_dim = 128,
        num_head = 8,
        num_encoder_layers = 6,
        num_decoder_layers = 6,
        d_feedforward = 2048,
        encoder_dropout = 0.1, 
        decoder_dropout = 0.05, 
        max_len = max_len,
        activation = 'relu'
    )
    model.to(device)
    logger.info("Model initialized.")
    tokenizer = Tokenizer(tok2idx = tok2id, idx2tok = id2tok, max_len = max_len)
    logger.info("Tokenizer initialized.")
    checkpoint = torch.load(ckpt_path, map_location = device, weights_only = True)
    model.load_state_dict(checkpoint['model_state'])
    logger.info("Model state loaded from %s", ckpt_path)

    # --- Load the training dataset ---
    assert os.path.exists(args.data), f"Training dataset not found at {args.data}"
    with open(args.data, 'r') as f:
        smiles = [line.strip() for line in f if line.strip()]
    logger.info(f"Training dataset loaded from {args.data} with {len(smiles)} molecules.")
    dataset = set(smiles) # Remove duplicates
    dataset = list(dataset)
    if not dataset: raise ValueError("Dataset is empty after loading and deduplication!")

    # --- Define the reward function ---
    rf = RewardFunction()

    if args.mode == 'test':
        run_unittest(
            model = model,
            tokenizer = tokenizer,
            rf = rf,
            dataset = dataset,
            device = device,
        )
    else:
        # --- Build the latent bank ---
        logger.info("Building the latent bank.")
        shape1, shape2 = build_bank(model = model,
                                    tokenizer = tokenizer,
                                    rf = rf,
                                    dataset = dataset,
                                    batch_size = args.batch_size if args.batch_size else 128,
                                    savepath = args.savepath if args.savepath else 'latent_bank.pt',
                                    device = device)
        logger.info(f"Latent bank shapes: Z: {shape1}, Y: {shape2}")
        logger.info("Latent bank construction completed.")
