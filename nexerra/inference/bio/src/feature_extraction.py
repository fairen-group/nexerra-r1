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
from typing import Iterable
import warnings as _warnings
import pandas as pd

from .constants import RDKIT_DEFAULT_INCLUDE_3D, RDKIT_DEFAULT_INCLUDE_FINGERPRINTS, RDKIT_FEATURES
from .rdkit_features import compute_rdkit_features
from .utils import validate_required_columns

# ----- Key input variables -----
DEFAULT_FEATURES: list[str] = RDKIT_FEATURES

def featurize_smiles_df(
    df: pd.DataFrame,
    smiles_col: str,
    features: Iterable[str] | None = None,
    include_fingerprints: bool | None = None,
    include_3d: bool | None = None,
    include_smiles: bool = False,
    smiles_id_col: str = "smiles",
    show_progress: bool = False,
    skip_invalid: bool = False,
    return_stats: bool = False,
) -> tuple[pd.DataFrame, list[list[str]]] | tuple[pd.DataFrame, list[list[str]], dict[str, object]]:
    '''Compute RDKit features for each SMILES in a DataFrame
    Returns (feature_df, warnings_per_row)
    ---
    include_smiles adds an identifier column with the original SMILES
    show_progress enables a tqdm progress bar
    skip_invalid skips SMILES that fail RDKit parsing and records counts
    return_stats returns a dict with skip/keep counts and example failures'''

    validate_required_columns(df.columns, [smiles_col], context = "featurize_smiles_df")
    features = list(features or DEFAULT_FEATURES)

    if include_fingerprints is None: include_fingerprints = RDKIT_DEFAULT_INCLUDE_FINGERPRINTS
    if include_3d is None: include_3d = RDKIT_DEFAULT_INCLUDE_3D

    if "DipoleMoment" in features and not include_3d: include_3d = True

    rows: list[dict[str, float | int | list[int]]] = []
    warnings_per_row: list[list[str]] = []
    kept_smiles: list[str] = []
    kept_indices: list[int] = []
    skipped: list[dict[str, str | int]] = []

    smiles_list = df[smiles_col].astype(str).tolist()
    iterator: Iterable[tuple[int, str]] = enumerate(smiles_list)
    if show_progress:
        from tqdm.auto import tqdm
        iterator = tqdm(list(enumerate(smiles_list)), desc = "RDKit features", leave = False)

    for idx, smi in iterator:
        try:
            feat, warn = compute_rdkit_features(
                smi,
                include_fingerprints = include_fingerprints,
                include_3d = include_3d)
        except Exception as e:
            if not skip_invalid: raise ValueError(f"Failed to featurize SMILES at row {idx}: {smi}") from e
            skipped.append({"row": idx, "smiles": str(smi), "error": f"{type(e).__name__}: {e}"})
            continue

        rows.append({k: feat[k] for k in features})
        warnings_per_row.append(warn)
        kept_smiles.append(str(smi))
        kept_indices.append(idx)

    if not rows: raise ValueError("No valid SMILES available for RDKit feature extraction.")

    if skip_invalid and skipped:
        total = len(smiles_list)
        skipped_count = len(skipped)
        skipped_pct = (skipped_count / total * 100.0) if total else 0.0
        examples = "; ".join(
            f"row {ex['row']} smiles={ex['smiles']} err={ex['error']}" for ex in skipped[:3]
        )
        _warnings.warn(
            f"Skipped {skipped_count}/{total} SMILES ({skipped_pct:.2f}%) during RDKit featurization. "
            f"Examples: {examples}"
        )

    feature_df = pd.DataFrame(rows)
    if include_smiles: feature_df.insert(0, smiles_id_col, kept_smiles)

    if return_stats:
        total = len(smiles_list)
        stats = {
            "total": total,
            "kept": len(rows),
            "skipped": len(skipped),
            "skipped_pct": (len(skipped) / total * 100.0) if total else 0.0,
            "kept_indices": kept_indices,
            "skipped_examples": skipped[:5],
        }
        return feature_df, warnings_per_row, stats
    return feature_df, warnings_per_row
