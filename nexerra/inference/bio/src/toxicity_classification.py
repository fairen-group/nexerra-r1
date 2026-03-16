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
import pandas as pd

from .constants import GHS_CATEGORY_LABELS, GHS_LD50_THRESHOLDS_MGKG, GHS_TOXIC_CATEGORIES
from .utils import validate_required_columns

# ----- Key input variables -----
DEFAULT_GHS_THRESHOLDS: list[float] = GHS_LD50_THRESHOLDS_MGKG

def ghs_category_from_ld50(ld50_mgkg: float, thresholds: Iterable[float] | None = None) -> int:
    '''Map LD50 (mg/kg) to a GHS acute toxicity category.
    Thresholds default to the GHS oral cutoffs and are used here as a proxy for IV/IP data'''

    if ld50_mgkg is None or pd.isna(ld50_mgkg):
        return 0
    thresholds = list(thresholds or DEFAULT_GHS_THRESHOLDS)
    if len(thresholds) != len(GHS_CATEGORY_LABELS):
        raise ValueError("Threshold count must match GHS_CATEGORY_LABELS length")
    for cutoff, label in zip(thresholds, GHS_CATEGORY_LABELS):
        if ld50_mgkg <= cutoff:
            return label
    return 0


def is_toxic_category(category: int | str | None) -> bool:
    '''Binary toxic flag based on numeric GHS category (1-4)'''
    if category is None: return False
    if isinstance(category, int): return category in GHS_TOXIC_CATEGORIES
    # Backward compatibility for string labels like "Category 1"
    try:
        import re
        m = re.search(r"(\\d+)", str(category))
        if not m: return False
        return int(m.group(1)) in GHS_TOXIC_CATEGORIES
    except Exception:
        return False


def classify_ghs(
    df: pd.DataFrame,
    ld50_col: str,
    category_col: str = "ghs_category",
    toxic_col: str = "ghs_toxic",
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    '''Add GHS category and binary toxic flag columns to a DataFrame'''
    validate_required_columns(df.columns, [ld50_col], context="classify_ghs")
    out = df.copy()
    out.loc[:, category_col] = out[ld50_col].apply(lambda x: ghs_category_from_ld50(x, thresholds))
    out.loc[:, toxic_col] = out[category_col].apply(is_toxic_category)
    return out
