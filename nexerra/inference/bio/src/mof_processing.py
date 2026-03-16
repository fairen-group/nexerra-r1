# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  This module implements a biocompatibility reward function compatible with the style
#  of Reward.py, while using a deterministic, auditable data pipeline.
#  Author: Ivan Zyuzin
#  --- 
#  
#  Please note: Parts of this code are still under-development [...]
#  MIT License
#  Copyright (c) 2026 The Authors
# ---------------------------------------

from __future__ import annotations
import pandas as pd
from .constants import MOF_PROPERTIES_COLS
from .utils import strip_lr_smiles, validate_required_columns

# ----- Key input variables -----
BRANCH_SMILES_COL: str = MOF_PROPERTIES_COLS["branch_smiles"]

def add_clean_linker_smiles(
    df: pd.DataFrame,
    output_col: str = "linker_smiles",
) -> pd.DataFrame:
    '''Create a cleaned linker SMILES column with [Lr] anchors removed'''
    validate_required_columns(df.columns, [BRANCH_SMILES_COL], context="add_clean_linker_smiles")
    out = df.copy()
    out.loc[:, output_col] = out[BRANCH_SMILES_COL].astype(str).apply(strip_lr_smiles)
    return out
