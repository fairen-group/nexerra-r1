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
from dataclasses import dataclass
import math
import re
from typing import Iterable
import pandas as pd

from .constants import METAL_TOXICITY_COLS
from .utils import normalize_minmax, validate_required_columns

# ----- Key input variables -----
LD50_REGEX = re.compile(r"([<>]?)\s*([0-9]+(?:\\.[0-9]+)?)")

@dataclass(frozen = True)
class MetalToxicityEntry:
    metal: str
    ld50_min: float | None
    ld50_max: float | None
    ld50_selected: float | None
    comparator: str | None
    confidence: str | None
    score: float | None

def _parse_toxicity_text(text: str | None) -> tuple[float | None, float | None, float | None, str | None]:
    '''Parse free-text toxicity field to extract LD50 values'''
    if text is None or not isinstance(text, str): return None, None, None, None
    matches = LD50_REGEX.findall(text)
    if not matches: return None, None, None, None
    comparators = [m[0] for m in matches if m[0]]
    comparator = comparators[0] if comparators else None
    values = [float(m[1]) for m in matches]
    if not values: return None, None, None, None
    ld50_min = min(values)
    ld50_max = max(values)
    # Conservative default -> use minimum value as the selected toxicity
    return ld50_min, ld50_max, ld50_min, comparator

def build_metal_toxicity_table(df: pd.DataFrame) -> dict[str, MetalToxicityEntry]:
    '''Parse metal toxicity data into a lookup table with normalized safety scores'''

    validate_required_columns(df.columns, METAL_TOXICITY_COLS.values(), context = "metal_toxicity")
    metals = []
    for _, row in df.iterrows():
        metal = str(row[METAL_TOXICITY_COLS["metal"]]).strip()
        tox_text = row[METAL_TOXICITY_COLS["toxicity"]]
        confidence = row[METAL_TOXICITY_COLS["confidence"]]
        ld50_min, ld50_max, ld50_sel, comparator = _parse_toxicity_text(tox_text)
        metals.append(
            {
                "metal": metal,
                "ld50_min": ld50_min,
                "ld50_max": ld50_max,
                "ld50_selected": ld50_sel,
                "comparator": comparator,
                "confidence": confidence,
            }
        )
    parsed = pd.DataFrame(metals)

    # Compute normalized scores on log10 scale
    valid = parsed["ld50_selected"].dropna()
    if valid.empty: raise ValueError("No parsable LD50 values in metal toxicity table")
    log_vals = valid.apply(lambda x: math.log10(float(x)))
    min_log = float(log_vals.min())
    max_log = float(log_vals.max())

    entries: dict[str, MetalToxicityEntry] = {}
    for _, row in parsed.iterrows():
        ld50_sel = row["ld50_selected"]
        score = None
        if ld50_sel is not None and not pd.isna(ld50_sel):
            score = normalize_minmax(math.log10(float(ld50_sel)), min_log, max_log)
        entry = MetalToxicityEntry(
            metal = row["metal"],
            ld50_min = row["ld50_min"],
            ld50_max = row["ld50_max"],
            ld50_selected = row["ld50_selected"],
            comparator = row["comparator"],
            confidence = row["confidence"],
            score = score,
        )
        entries[entry.metal] = entry
    return entries


def metal_safety_score(
    metal_symbol: str | None,
    table: dict[str, MetalToxicityEntry],
    neutral_score: float = 0.5,
) -> tuple[float, str | None]:
    '''Look up a metal safety score. Returns (score, warning)'''
    if metal_symbol is None:
        return neutral_score, "Metal symbol missing; using neutral metal safety score"
    key = str(metal_symbol).strip()
    entry = table.get(key)
    if entry is None or entry.score is None:
        return neutral_score, f"Metal '{metal_symbol}' not found or missing LD50; using neutral score"
    return float(entry.score), None
