# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Dataloader module for the Transformer VAE.
# All the heavy lifting is done in utils/preprocess.py 
# Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import Dataset
import pickle

class TrainDataLoader(Dataset):
    '''DataLoader for the training loop.'''
    def __init__(self, dataset_path):
        super().__init__()
        with open(dataset_path, 'rb') as f: self.data = pickle.load(f)
    def __getitem__(self, idx):
        return{
            "input": self.data["encoded_dataset"][idx],
            "target": self.data["encoded_dataset"][idx],
        }
    
    def __len__(self):
        return len(self.data["encoded_dataset"])    