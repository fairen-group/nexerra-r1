# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  Trainer module for the Transformer VAE. 
#  Example usage: 
#  
#   1. Train from beginning 
#   python Trainer.py --data [path_to_tokenized_dataset] --rs [path_to_raw_smiles (.csv)] --batch [128] /
#   --ckpt [path_to_save_ckpt_directory] 
#   2. Resume training from ckpt
#   python Trainer.py --data [path_to_tokenized_dataset] --rs [path_to_raw_smiles (.csv)] --batch [128] /
#   --ckpt [path_to_save_ckpt_directory] --resume [path_to_saved_ckpt] --epoch [epoch_number]
#   3. To ensure shapes are correct: enable shape_flag; --shape_flag True  
#  
#  Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Standard library imports ---
import pandas as pd
import logging
import argparse
from collections import deque
import numpy as np
import pickle
from tqdm import tqdm
import random
random.seed(42)

# --- RDKit imports ---
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit import RDLogger

# --- PyTorch imports ---
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as DL 
from torch.utils.data import random_split
from torch.optim import AdamW

# --- Local imports ---
from nexerra.model.HTVAE import VAEModel
from nexerra.utils.tokenizer import Tokenizer
from nexerra.utils.schedulers import LinearBetaScheduler, DropoutScheduler, WarmupCosineScheduler, CyclicalBetaScheduler, RampBetaScheduler
from nexerra.utils.masks import generate_causal_mask, generate_key_padding_mask, WordDropout
from nexerra.model.DataLoader import TrainDataLoader

# -------------------------------------------
# CSV logging configuration
# -------------------------------------------
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'logs'))
import csv 
csv_log_path = "metrics.csv"
csv_fieldnames = ["epoch", "loss", "rec_loss","kl_loss", "active_units", "token_acc", "recon_acc", "int_div", "validity", "novelty"]
if not os.path.exists(os.path.join(log_dir, csv_log_path)):
    with open(os.path.join(log_dir, csv_log_path), mode = "w", newline = "") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = csv_fieldnames)
        writer.writeheader()

# -------------------------------------------
# Logging configuration: Step-level loggers for better tracking
# -------------------------------------------
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

step_logger = logging.getLogger("step_logger")
step_handler = logging.FileHandler(os.path.join(log_dir, "step_metrics.txt"))
step_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
step_logger.addHandler(step_handler)

smi_logger = logging.getLogger("smi_logger")
smi_handler = logging.FileHandler(os.path.join(log_dir, "gen_smi.txt"))
smi_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
smi_logger.addHandler(smi_handler)
smi_logger.propagate = False

raw_dump = logging.getLogger("raw_smi_logger")
raw_dump_handler = logging.FileHandler(os.path.join(log_dir, "raw_smi.txt"))
raw_dump.addHandler(raw_dump_handler)
raw_dump.propagate = False

recon_logger = logging.getLogger("recon_logger")
recon_handler = logging.FileHandler(os.path.join(log_dir, "recon_smi.txt"))
recon_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
recon_logger.addHandler(recon_handler)
recon_logger.propagate = False

recon_dump = logging.getLogger("recon_dump_logger")
recon_dump_handler = logging.FileHandler(os.path.join(log_dir, "recon_dump.txt"))
recon_dump.addHandler(recon_dump_handler)
recon_dump.propagate = False

RDLogger.DisableLog('rdApp.*') # Disable RDKit warnings

import pyfiglet
def display_banner():
    banner = pyfiglet.figlet_format("Nexerra", font="slant")
    print(banner)

# --- Utility functions ---
def sanitize_lr(smi: str) -> str:
    '''Sanitize SMILES by replacing Lr with [*] (dummy atom)'''
    return smi.replace('[Lr]', '[*]')

def int_div(smiles_list: list, radius: int = 2, nBits: int = 2048) -> float:
    '''Calculates the internal diversity of the list of SMILES strings passed'''
    fingerprints = []
    # ensure smiles_list has length >= 2
    for smi in smiles_list:
        try:
            smi2 = sanitize_lr(smi)
            mol = Chem.MolFromSmiles(smi2)
            if mol is None: continue 
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = radius, nBits = nBits)
            fingerprints.append(fp)
        except Exception as e:
            logger.warning(f"Error processing SMILES {smi}: {e}")
            continue
    
    N = len(fingerprints)
    if N < 2: return 0.0
    
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            similarity = TanimotoSimilarity(fingerprints[i], fingerprints[j])
            dists.append(1 - similarity)  # convert similarity to distance
    
    return float(np.mean(dists)) if dists else 0.0


# --- Main trainer module ---
class VAETrainer:
    def __init__(
            self,
            model: nn.Module,
            data_path: str,
            raw_smiles_list: list[str],
            tokenizer: Tokenizer,
            tok2id: dict,
            id2tok: dict,
            max_len: int,
            device: torch.device,
            config: dict):
      
      self.device = device
      self.config = config
      self.tokenizer = tokenizer
      self.id2tok = id2tok
      self.tok2id = tok2id
      self.max_len = max_len
      self.raw_smiles_list = raw_smiles_list
      
      # -------------------------------
      # word dropout (optional);
      # only implement if you want to weaken the decoders ability to reconstruct the input
      # -------------------------------

      '''
      self.word_dropout = WordDropout(pad_token_id = self.config['pad_index'], unk_token_id = self.config['unk_index'])
      self.word_dropout_scheduler = DropoutScheduler(decay_start = self.config['word_dropout_decay_start'],
                                                        start_value = self.config['word_dropout_init'],
                                                        end_value = self.config['word_dropout_final'],
                                                        total_steps = self.config['word_dropout_steps'])
      '''

      # --- load & split dataset (90/5/5 split) ---
      self.data_path = data_path
      dataset = TrainDataLoader(self.data_path)
      total_data = len(dataset)
      val_size = int(0.05 * total_data)
      test_size = int(0.05 * total_data)
      train_size = total_data - val_size - test_size
      train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size]) 
      logger.info(f"Train dataset size: {len(train_dataset)} | Validation dataset size: {len(val_dataset)} | Test dataset size: {len(test_dataset)}")

      self.train_loader = DL(train_dataset, batch_size = config["batch_size"], shuffle = True, drop_last = True)
      self.val_loader = DL(val_dataset, batch_size = config["batch_size"], shuffle = False)
      self.test_loader = DL(test_dataset, batch_size = config["batch_size"], shuffle = False)
      
      # --- store SMILES for novelty/reconstruction evaluation ---
      self.train_smiles_set = set([raw_smiles_list[i] for i in train_dataset.indices])
      self.val_smiles = [raw_smiles_list[i] for i in val_dataset.indices]
      self.test_smiles = [raw_smiles_list[i] for i in test_dataset.indices]
      if self.config['save_smiles_flag'] == True:
        with open('train_smiles.txt', 'w') as f:
            for smi in self.train_smiles_set: f.write(f"{smi}\n")
        with open('val_smiles.txt', 'w') as f:
            for smi in self.val_smiles: f.write(f"{smi}\n")
        with open('test_smiles.txt', 'w') as f:
            for smi in self.test_smiles: f.write(f"{smi}\n")

      # --- others ---  
      self.epochs = config['epochs']
      self.total_steps = len(self.train_loader) * self.epochs
      self.gradient_clip = config.get('gradient_clip', 1.0)
      self.global_step = 0
      
      # --- Depending on the beta scheduler, you many need one of the following ---
      self.steps_per_epoch = len(self.train_loader)
      self.warmup_steps = self.steps_per_epoch * config['warmup_epochs']
      # self.cycle_length = int((config['epochs'] / 4) * self.steps_per_epoch) # 4 cycles over the training
      
      # --- model, optimizer, scheduler(s) --- 
      self.model = model.to(device)
      # self.max_beta = float(1 / config['latent_dim']) # use other one if fixed max_beta needed
      self.max_beta = self.config['max_beta']
      self.all_params = list(self.model.parameters())
      self.optimizer = AdamW(self.all_params, lr = config['lr'])
      self.scheduler = WarmupCosineScheduler(self.optimizer, warmup_steps = config['lr_warmup_steps'], 
                                                total_steps = self.total_steps, min_lr = 5e-5) 
      
      # --- Initialize beta scheduler (pick your poison) ---
      # --- 1. Linear Beta Scheduler ---
      self.beta_scheduler = LinearBetaScheduler(warmup_steps = self.warmup_steps, max_beta = self.max_beta)
      # --- 2. Cyclical Beta Scheduler ---
      # self.beta_scheduler = CyclicalBetaScheduler(cycle_length = self.cycle_length, max_beta = self.max_beta)
      # --- 3. Piecewise Linear Beta Scheduler ---
      # self.beta_scheduler = RampBetaScheduler(warmup_steps = self.steps_per_epoch * 2,
      #                                          ramp_steps = self.steps_per_epoch * 10,
      #                                          max_beta = self.max_beta)

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        '''Compute the KL divergence per sample; returns mean KL over batch'''
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = torch.clamp(kl_per_dim, min = self.config['free_bits']) # clamp to free bits to force minimal information passing through the bottleneck
        kl_loss = kl_per_dim.sum(dim = 1).mean()
        return kl_loss

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = total_rec = total_kl = 0.0
        kl_raw_window = deque(maxlen = 100)
        # ---- reconstruction accuracy, validity, novelty windows ----
        recon_window = deque(maxlen = 3)
        intdiv_window = deque(maxlen = 3)
        validity_window = deque(maxlen = 3)
        novelty_window = deque(maxlen = 3)
        # --- logs ---
        active_units_log = []
        token_acc_log = []
        recon_acc_log = []
        int_div_log = []
        validity_log = []
        novelty_log = []

        pbar = tqdm(self.train_loader, desc = f"Epoch {epoch + 1} / {self.epochs}")
        
        for batch in pbar:
            src = batch['input'].to(self.device)
            tgt = batch['target'].to(self.device)
            
            src_key_pad_mask = generate_key_padding_mask(src, pad_index = self.model.pad_index).to(self.device)
            tgt_key_pad_mask = generate_key_padding_mask(tgt, pad_index = self.model.pad_index).to(self.device)
            causal_mask = generate_causal_mask(tgt.size(1)).to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            # --- Word dropout (optional) ---
            '''
            self.word_dropout.train()
            dropout_prob = self.word_dropout_scheduler.get_dropout(self.global_step)
            tgt_input_dropped = self.word_dropout(dropout_prob = dropout_prob, input_ids = tgt_input)
            '''

            # --- Forward pass ---
            logits, z, mu, logvar = self.model(
                src = src, 
                tgt = tgt_input, # if dropout --> use tgt_input_dropped
                src_mask = None,
                src_key_padding_mask = src_key_pad_mask,
                tgt_mask = causal_mask[:-1, :-1],
                tgt_key_padding_mask = tgt_key_pad_mask[:, :-1])
            
            # --- Reconstruction loss --- 
            rec_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_labels.contiguous().reshape(-1),
                ignore_index = self.model.pad_index, 
            )

            # --- Per-token accuracy ---
            with torch.no_grad():
                preds = logits.argmax(dim = -1)
                mask = (tgt_labels != self.model.pad_index)
                correct = (preds == tgt_labels) & mask
                token_acc = (correct.sum().item() / mask.sum().item()) * 100
                token_acc_log.append(token_acc)
            kl = self.kl_loss(mu, logvar)
            # --- Track raw KL loss for moving average --- 
            kl_raw = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1).mean()
            kl_raw_window.append(kl_raw.detach().item())
            
            # --- Beta annealing ---
            beta = self.beta_scheduler.get_beta(self.global_step)

            # --- soft ELBO and backward pass ---
            loss = rec_loss + beta * kl
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, self.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.detach().item()
            total_rec += rec_loss.detach().item()
            total_kl += kl.detach().item()
            
            # --- Active dim count --- 
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_mean_dim = kl_per_dim.mean(dim = 0)
            active_dims = (kl_mean_dim > 0.01).sum().item()

            # --- Active units in latent space --- 
            with torch.no_grad():
                var_dim = mu.var(dim = 0)
                active = (var_dim > 0.01).sum().item()
                active_units_log.append(active)
            
            # --- periodic evaluation ---
            if self.global_step % 1000 == 0 and self.global_step > 0:
                
                smi_logger.info(f"Step {self.global_step}")
                recon, intdiv, val, nov, neos = self._eval_gen_and_recon(num_samples = 100, global_step = self.global_step)
                self.model.train()
                if neos > 0: logger.warning(f"Number of reconstructed SMILES without EOS: {neos}")

                # ---- Log the recon, validity, novelty metrics ----
                recon_acc_log.append(recon)
                int_div_log.append(intdiv)
                validity_log.append(val)
                novelty_log.append(nov)

                # --- moving average of recon, validity, novelty ---
                recon_window.append(recon)
                recon_ma = sum(recon_window) / len(recon_window)
                intdiv_window.append(intdiv)
                int_div_ma = sum(intdiv_window) / len(intdiv_window)
                validity_window.append(val)
                val_ma = sum(validity_window) / len(validity_window)
                novelty_window.append(nov)
                nov_ma = sum(novelty_window) / len(novelty_window)
            
                step_logger.info(
                    f"Step {self.global_step} | Loss: {loss.detach().item():.3f}, |"
                    f"Rec: {rec_loss.detach().item():.3f}, | KL: {kl.detach().item():.3f}, |"
                    f"Beta: {beta:.3f} | Active Dims: {active_dims} |"
                    f"Recon (MA): {recon_ma:.2f} | IntDiv (MA): {int_div_ma: 2f} | Validity (MA): {val_ma:.2f}, | Novelty (MA): {nov_ma:.2f}"
                )
                
            pbar.set_postfix({
                'loss': loss.detach().item(),
                'rec_loss': rec_loss.detach().item(),
                'token_acc': f"{token_acc:.2f}",
                'kl_loss': kl.detach().item(),
                'beta': f"{beta:.3f}",
                'active_units': active,
                'recon_acc': np.mean(recon_acc_log),
                'validity': np.mean(validity_log),
                'novelty': np.mean(novelty_log)
            })

            self.global_step += 1
        
        return {
            'loss': total_loss / len(self.train_loader),
            'rec_loss': total_rec / len(self.train_loader),
            'token_acc': np.mean(token_acc_log),
            'kl_loss': total_kl / len(self.train_loader),
            'active_units': np.mean(active_units_log),
            'recon_acc': np.mean(recon_acc_log),
            'int_div': np.mean(int_div_log),
            'validity': np.mean(validity_log),
            'novelty': np.mean(novelty_log)
        }

    @torch.no_grad()
    def _eval_gen_and_recon(self, num_samples: int = 100, global_step: int = 0):
        self.model.eval()
        # self.word_dropout.eval()
        device = self.device
        eval_n = min(num_samples, len(self.val_smiles))
        # --- Sample and generate ---
        z = torch.randn(num_samples, self.config["latent_dim"], device = device)
        gen_tokens = self.model.generate(latent_variable = z, max_len = self.max_len, temperature = self.config['temperature'], 
                                         top_p_val = self.config['top_p'])
        # --- validity and novelty ---
        valid = novel = 0
        gen_smiles = []
        for seq in gen_tokens.cpu().tolist():
            smi = self.tokenizer.decode_single(seq)
            raw_dump.info(f"Raw SMILES: {smi}\n")
            try:
                if smi is not None and len(smi) > 0:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        Chem.rdmolops.SanitizeMol(mol)
                        valid += 1
                        gen_smiles.append(smi)
                        if smi not in self.train_smiles_set:
                            novel += 1
                            smi_logger.info(f"Generated novel SMILES: {smi}")
            except Exception: pass
        
        if len(gen_smiles) <= 2: int_div_val = 0.0
        else:
            int_div_val = int_div(gen_smiles, radius = 2, nBits = 2048)
        validity = (valid / len(gen_tokens)) * 100
        novelty = (novel / valid) * 100 if valid > 0 else 0.0

        # --- reconstruction accuracy ---
        recon_dump.info(f"Step: {global_step}\n")
        recon_logger.info(f"Step: {global_step}\n")
        recon_correct = recon_total = 0
        sample_smiles = random.sample(self.val_smiles, eval_n)
        no_eos = 0
        for smi in tqdm(sample_smiles):
            tokens = self.tokenizer.encode_single(smi)
            tokens_list = tokens.tolist()
            eos_index = self.tok2id["[EOS]"]
            L = tokens_list.index(eos_index) + 1 if eos_index in tokens_list else len(tokens_list)
            tokens = torch.tensor(tokens[: L], device = device).unsqueeze(0)
            
            target_input = tokens[:, :-1]
            pad_id = self.model.pad_index
            src_key_padding_mask = (tokens == pad_id).to(self.device)
            target_key_padding_mask = (target_input == pad_id).to(self.device)
            L1 = target_input.size(1)
            target_mask = torch.triu(torch.ones(L1, L1, device = self.device), 1).bool()
            
            logits, _, _, _ = self.model(
                                    src = tokens, 
                                    tgt = target_input,
                                    src_mask = None, 
                                    src_key_padding_mask = src_key_padding_mask, 
                                    tgt_key_padding_mask = target_key_padding_mask, 
                                    tgt_mask = target_mask)
            pred_ids = logits.argmax(-1).squeeze(0).cpu()
            pred_ids = pred_ids.tolist()
            if eos_index in pred_ids:
                pred_ids = pred_ids[: pred_ids.index(eos_index) + 1]
            else:
                no_eos += 1
                pred_ids.append(eos_index)
            pred_ids = torch.tensor(pred_ids, device = device).unsqueeze(0)
            recon_smi = self.tokenizer.decode_single(pred_ids.squeeze(0))
            recon_dump.info(f"Original SMILES:  {smi}; Reconstructed SMILES: {recon_smi}\n")
            
            mol_pred = Chem.MolFromSmiles(recon_smi)
            mol_true = Chem.MolFromSmiles(smi)
            if mol_pred and mol_true:
                if (Chem.MolToSmiles(mol_pred, canonical=True) == Chem.MolToSmiles(mol_true, canonical=True)):
                    recon_correct += 1
                    recon_logger.info(f"Input SMILES: {smi}; Recon SMILES: {recon_smi}\n")
            recon_total += 1

        recon_accuracy = (recon_correct / recon_total) * 100 if recon_total > 0 else 0.0
        
        return recon_accuracy, int_div_val, validity, novelty, no_eos

# ------------------------------------------
# Main Function
# ------------------------------------------

if __name__ == "__main__":
    display_banner()
    parser = argparse.ArgumentParser(description='Training loop')
    parser.add_argument('--data', type=str, required=True, help='Path to tokenized dataset')
    parser.add_argument('--rs', type=str, required=True, help='Path to raw smiles')
    parser.add_argument('--batch', type=str, required=True, help='Batch size')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to ckpt directory')
    parser.add_argument('--shape_flag', type=str, default = 0, help='Check shapes (yes: 1, no: 0)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch to resume training from' )
    parser.add_argument('--save_smiles_flag', action='store_true', help='Save smiles flag (True/False)')
    args = parser.parse_args()
    
    # ----- setup device -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ----- Load the dataset & dataloader -----
    data_path = args.data
    raw_smiles = args.rs
    batch_size = int(args.batch)
    ckpt_dir = args.ckpt

    with open(data_path, 'rb') as f: dataset = pickle.load(f)
    logger.info(f"Loaded dataset from {data_path}")
    data = dataset['encoded_dataset']
    tok2id = dataset['tok2id']
    id2tok = dataset['id2tok']
    vocab_size = dataset['vocab_size']
    max_len = dataset['max_len']
    logger.info(f"Max sequence length: {max_len}")
    sos_index = dataset['sos_index']
    eos_index = dataset['eos_index']
    padding_index = dataset['padding_index']
    unk_index = dataset['unk_index']
    
    raw_smiles_list = pd.read_csv(raw_smiles)['smiles'].tolist()
    logger.info(f"Loaded {len(raw_smiles_list)} SMILES")
    
    # ----- Trainer & config -----
    config = {
        'lr': 1e-4,
        'lr_warmup_steps': 200000,
        'latent_dim': 128,
        'batch_size': batch_size,
        'warmup_epochs': 60, # <-- increase to 100 if you want even better stability.
        'max_beta': 0.01,
        'epochs': 120, # Production model training should be 120 epochs.
        'free_bits': 0.05, # Prevent KL collapse by clamping to free bits.
        'gradient_clip': 1.0,
        'word_dropout_init': 0.15,
        'word_dropout_final': 0.0,
        'word_dropout_decay_start': 6000,
        'word_dropout_steps': 50000,
        'temperature': 0.8,
        'top_p': 0.9,
        'save_smiles_flag' : args.save_smiles_flag,
    }

    # ----- Model initialization -----
    model = VAEModel(
        device = device,
        shape_flag = int(args.shape_flag),
        vocab_size = vocab_size,
        sos_index = sos_index,
        eos_index = eos_index,
        pad_index = padding_index,
        unk_index = unk_index,
        d_model = 512,
        latent_dim = config['latent_dim'],
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

    tokenizer = Tokenizer(tok2idx = tok2id, idx2tok = id2tok, max_len=max_len)
    trainer = VAETrainer(model, data_path, raw_smiles_list, tokenizer, tok2id, id2tok, max_len, device, config)

    if args.resume is not None:
        assert args.epoch > 0, "The epoch must be provided with the resume flag"
        start_epoch = int(args.epoch) 
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state'])
        trainer.global_step = checkpoint.get('global_step', 0)
        # Optionally, parse epoch from filename or store in checkpoint
        print(f"Resumed from checkpoint {args.resume} at global step {trainer.global_step}")
    else: start_epoch = 0
    
    # ----- Training loop -----
    for epoch in range(start_epoch, config['epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['epochs']}")
        metrics = trainer.train_epoch(epoch)
        # --- Log metrics to CSV ---
        with open(os.path.join(log_dir, csv_log_path), mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writerow({
                'epoch': epoch + 1,
                'loss': metrics['loss'],
                'rec_loss': metrics['rec_loss'],
                'kl_loss': metrics['kl_loss'],
                'active_units': metrics['active_units'],
                'token_acc': metrics['token_acc'],
                'recon_acc': metrics['recon_acc'],
                'int_div': metrics['int_div'],
                'validity': metrics['validity'],
                'novelty': metrics['novelty']
            })
    
        ckpt_path = os.path.join(ckpt_dir, f"vae_epoch_{epoch + 1}.pt")
        torch.save({
            'model_state': model.state_dict(), 
            'optimizer_state': trainer.optimizer.state_dict(),
            'scheduler_state': trainer.scheduler.state_dict(),
            'global_step': trainer.global_step},
            ckpt_path)
        logger.info(f"Model checkpoint saved at {ckpt_path}.")
