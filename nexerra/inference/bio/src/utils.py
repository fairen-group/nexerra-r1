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
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import subprocess
from typing import Iterable, Sequence
import numpy as np
from rdkit import Chem

def ensure_numpy_rec() -> None:
    '''Some libraries access np.rec, which triggers a lazy import of numpy.rec.
    NumPy does not ship that submodule in all versions, so we provide a
    compatibility alias to numpy.core.records when needed'''
    try: import numpy.rec as _rec; return
    except Exception:
        try: import numpy.core.records as _rec
        except Exception as exc: raise ImportError("NumPy core.records is unavailable for recarray compatibility.") from exc
        np.rec = _rec
        import sys
        sys.modules.setdefault("numpy.rec", _rec)


def clamp01(x: float) -> float:
    '''Clamp a numeric value into [0, 1]'''
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return float(x)


def zlike_window(x: float, lo: float, hi: float) -> float:
    '''Soft preference window:
    1.0 inside [lo, hi], linearly decays outside'''
    
    if lo >= hi: raise ValueError("lo must be < hi")
    if x < lo: return clamp01(1.0 - (lo - x) / lo) if lo != 0 else 0.0
    if x > hi: return clamp01(1.0 - (x - hi) / hi) if hi != 0 else 0.0
    return 1.0

def normalize_minmax(x: float, min_val: float, max_val: float) -> float:
    '''Min-max scale to [0, 1], clamped'''
    if max_val <= min_val: raise ValueError("max_val must be greater than min_val")
    return clamp01((x - min_val) / (max_val - min_val))


def strip_lr_smiles(smiles: str) -> str:
    '''Remove [Lr] linker anchor tokens from SMILES.
    The result is intended for RDKit descriptor calculations on the organic linker
    
    # -------------
    # One comment here: this will fail if [Lr] is inside a ring, 
    # better to replace with [*], [C] or [N]
    # -------------
    if smiles is None: raise ValueError("SMILES is None")
    import re
    # Remove Lr anchor atoms (including any bracketed annotations)
    cleaned = re.sub(r"\[Lr[^\]]*\]", "", smiles)
    while "()" in cleaned: cleaned = cleaned.replace("()", "")

    # Collapse dot separators and trim leading/trailing dots
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = cleaned.strip(".")
    return cleaned
    '''

    ''' Anchor - anchor length depends on 3D embedding, which could give issues if the [Lr] anchor is present
        replace each [Lr] with a chemically reasonable benign Carbon
        - convert anchor --> carbon [in place, i.e., indices preserved];
        Returns
        ---
        - new_mol
        - anchor_idxs: indices where anchors were found (now carbon)'''
    anchor_symbol = "Lr"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Failed to convert SMILES: {smiles} --> mol")
    rw = Chem.RWMol(mol)
    # User may give molecules with -COOH or -N or -SO3H directly to the reward function [..]
    anchor_idxs = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == anchor_symbol]
    
    # convert anchors into benign carbons so RDKit embedding doesnt fail
    for idx in anchor_idxs:
        a = rw.GetAtomWithIdx(idx)
        if a.IsInRing():
            a.SetAtomicNum(7) # N
        else:
            a.SetAtomicNum(6) # C
        a.SetFormalCharge(0)
        a.SetNoImplicit(False)
        a.SetIsotope(0)

    new_mol = rw.GetMol()
    Chem.SanitizeMol(new_mol)
    cleaned = Chem.MolToSmiles(new_mol, canonical = True)
    return cleaned


def validate_required_columns(
    columns: Iterable[str],
    required: Sequence[str],
    context: str,
) -> None:
    '''Validate required columns exist in a dataset'''
    col_set = set(columns)
    missing = [c for c in required if c not in col_set]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def compute_file_sha256(path: Path) -> str:
    '''Compute SHA-256 hash of a file'''
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


@dataclass(frozen = True)
class ProvenanceRecord:
    created_utc: str
    sources: dict[str, str]
    code_version: str | None


def build_provenance_record(source_paths: Sequence[Path], code_version: str | None) -> ProvenanceRecord:
    '''Create a provenance record containing hashes for source files'''
    sources = {str(p): compute_file_sha256(p) for p in source_paths}
    created = datetime.now(timezone.utc).isoformat()
    return ProvenanceRecord(created_utc = created, sources = sources, code_version = code_version)


def write_provenance_json(path: Path, record: ProvenanceRecord) -> None:
    '''Write provenance info to a JSON file'''
    import json
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_utc": record.created_utc,
                "sources": record.sources,
                "code_version": record.code_version,
            },
            f,
            indent = 2,
        )


def get_git_commit(project_root: Path) -> str | None:
    '''Return the current git commit hash if available'''
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check = True,
            capture_output = True,
            text = True,)
        return result.stdout.strip()
    except Exception:
        return None


def init_run_log(logs_dir: Path, run_name: str) -> Path:
    '''Create a timestamped log file path under logs_dir'''
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{run_name}_{ts}.log"
