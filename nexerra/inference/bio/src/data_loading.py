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

from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import (
    ACUTE_TOX_COLS,
    ACUTE_TOX_IP_PATH,
    ACUTE_TOX_IV_PATH,
    METAL_TOXICITY_COLS,
    MOF_METAL_TOXICITY_PATH,
    MOF_PROPERTIES_COLS,
    MOF_SMILES_PATH,
)
from .utils import ensure_numpy_rec, validate_required_columns

# ----- Key inputs (paths) -----
ACUTE_TOX_IV_SOURCE: Path = ACUTE_TOX_IV_PATH
ACUTE_TOX_IP_SOURCE: Path = ACUTE_TOX_IP_PATH
MOF_PROPERTIES_SOURCE: Path = MOF_SMILES_PATH
MOF_METAL_TOXICITY_SOURCE: Path = MOF_METAL_TOXICITY_PATH

def _read_csv(
    path: Path,
    dtype: dict[str, str],
    required_cols: Iterable[str],
    context: str,
) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(f"Missing source file: {path}")
    ensure_numpy_rec()
    df = pd.read_csv(path, dtype = dtype)
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    validate_required_columns(df.columns, list(required_cols), context = context)
    if df.shape[0] == 0: raise ValueError(f"{context} loaded with zero rows")
    return df

def load_acute_tox_iv(path: Path | None = None) -> pd.DataFrame:
    '''Load the intravenous acute toxicity dataset with explicit dtypes'''
    path = path or ACUTE_TOX_IV_SOURCE
    dtype = {
        ACUTE_TOX_COLS["taid"]: "string",
        ACUTE_TOX_COLS["name"]: "string",
        ACUTE_TOX_COLS["iupac_name"]: "string",
        ACUTE_TOX_COLS["pubchem_cid"]: "Int64",
        ACUTE_TOX_COLS["canonical_smiles"]: "string",
        ACUTE_TOX_COLS["inchikey"]: "string",
        ACUTE_TOX_COLS["tox_value"]: "float64",
    }
    return _read_csv(path, dtype, ACUTE_TOX_COLS.values(), context = "acute_tox_iv")


def load_acute_tox_ip(path: Path | None = None) -> pd.DataFrame:
    '''Load the intraperitoneal acute toxicity dataset with explicit dtypes'''
    path = path or ACUTE_TOX_IP_SOURCE
    dtype = {
        ACUTE_TOX_COLS["taid"]: "string",
        ACUTE_TOX_COLS["name"]: "string",
        ACUTE_TOX_COLS["iupac_name"]: "string",
        ACUTE_TOX_COLS["pubchem_cid"]: "Int64",
        ACUTE_TOX_COLS["canonical_smiles"]: "string",
        ACUTE_TOX_COLS["inchikey"]: "string",
        ACUTE_TOX_COLS["tox_value"]: "float64",
    }
    return _read_csv(path, dtype, ACUTE_TOX_COLS.values(), context = "acute_tox_ip")


def load_mof_properties(path: Path | None = None) -> pd.DataFrame:
    '''
    Load linker/MOF input data with graceful handling of minimal schemas.
    Required:
    - `smiles` column
    ---
    Optional:
    - MOF columns such as agsa/pld/lcd/density/organic_core/metal_node/topology.
      Missing optional columns are added as NA/NaN so downstream code can run'''
    
    path = path or MOF_PROPERTIES_SOURCE
    if not path.exists(): raise FileNotFoundError(f"Missing source file: {path}")

    ensure_numpy_rec()
    df = pd.read_csv(path)
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    if df.shape[0] == 0: raise ValueError("mof_properties loaded with zero rows")

    canonical_smiles_col = MOF_PROPERTIES_COLS["branch_smiles"]
    validate_required_columns(
        df.columns,
        [canonical_smiles_col],
        context = "mof_properties",
    )

    string_cols = [
        MOF_PROPERTIES_COLS["organic_core"],
        MOF_PROPERTIES_COLS["metal_node"],
        MOF_PROPERTIES_COLS["topology"],
        canonical_smiles_col,
    ]
    numeric_cols = [
        MOF_PROPERTIES_COLS["lcd"],
        MOF_PROPERTIES_COLS["pld"],
        MOF_PROPERTIES_COLS["density"],
        MOF_PROPERTIES_COLS["agsa"],
    ]

    for col in string_cols:
        if col not in df.columns: df.loc[:, col] = pd.Series([pd.NA] * len(df), dtype="string")
        else: df.loc[:, col] = df[col].astype("string")

    for col in numeric_cols:
        if col not in df.columns: df.loc[:, col] = float("nan")
        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def load_metal_toxicity(path: Path | None = None) -> pd.DataFrame:
    '''Load the metal toxicity lookup table with explicit dtypes'''
    path = path or MOF_METAL_TOXICITY_SOURCE
    dtype = {
        METAL_TOXICITY_COLS["metal"]: "string",
        METAL_TOXICITY_COLS["toxicity"]: "string",
        METAL_TOXICITY_COLS["reference"]: "string",
        METAL_TOXICITY_COLS["confidence"]: "string",
    }
    return _read_csv(path, dtype, METAL_TOXICITY_COLS.values(), context="metal_toxicity")
