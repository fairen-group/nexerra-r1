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
import pandas as pd

from .constants import ACUTE_TOX_COLS, ACUTE_TOX_IV_PATH, INTERIM_DIR
from .data_loading import load_acute_tox_iv
from .toxicity_classification import classify_ghs
from .utils import build_provenance_record, get_git_commit, validate_required_columns, write_provenance_json

# ----- Key input variables -----
ACUTE_TOX_SOURCE: Path = ACUTE_TOX_IV_PATH
DEFAULT_OUTPUT_PATH: Path = INTERIM_DIR / "acute_tox_iv_ld50.parquet"
DEFAULT_METADATA_PATH: Path = INTERIM_DIR / "acute_tox_iv_ld50.provenance.json"
DEFAULT_LABELED_PATH: Path = INTERIM_DIR / "acute_tox_iv_labeled.parquet"
DEFAULT_LABELED_METADATA_PATH: Path = INTERIM_DIR / "acute_tox_iv_labeled.provenance.json"

def create_acute_tox_ld50_df(
    source_path: Path | None = None,
    save_path: Path | None = None,
    metadata_path: Path | None = None,
    code_version: str | None = None,
) -> pd.DataFrame:
    '''Load the IV acute toxicity dataset and optionally save a cleaned copy'''

    df = load_acute_tox_iv(source_path)
    validate_required_columns(df.columns, ACUTE_TOX_COLS.values(), context = "create_acute_tox_ld50_df")
    if save_path is not None:
        save_path.parent.mkdir(parents = True, exist_ok = True)
        df.to_parquet(save_path, index = False)

        commit = code_version or get_git_commit(Path(__file__).resolve().parents[1])
        record = build_provenance_record([source_path or ACUTE_TOX_SOURCE], code_version=commit)
        write_provenance_json(metadata_path or DEFAULT_METADATA_PATH, record)
    return df


def create_acute_tox_labeled_df(
    source_path: Path | None = None,
    save_path: Path | None = None,
    metadata_path: Path | None = None,
    code_version: str | None = None,
) -> pd.DataFrame:
    '''Load the IV acute toxicity dataset, add GHS labels, and optionally save a labeled copy'''
    
    df = load_acute_tox_iv(source_path)
    validate_required_columns(df.columns, ACUTE_TOX_COLS.values(), context = "create_acute_tox_labeled_df")
    labeled = classify_ghs(df, ld50_col = ACUTE_TOX_COLS["tox_value"])

    if save_path is not None:
        save_path.parent.mkdir(parents = True, exist_ok = True)
        labeled.to_parquet(save_path, index = False)

        commit = code_version or get_git_commit(Path(__file__).resolve().parents[1])
        record = build_provenance_record([source_path or ACUTE_TOX_SOURCE], code_version=commit)
        write_provenance_json(metadata_path or DEFAULT_LABELED_METADATA_PATH, record)
    return labeled
