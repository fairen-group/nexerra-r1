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
#  MIT License
#  Copyright (c) 2026 The Authors
# ---------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings as _warnings

import numpy as np
import pandas as pd

from .constants import (
    GHS_TOXIC_CATEGORIES,
    TOX_SPACE_NUMERIC_FEATURES,
    TOX_FRAGMENT_MAX_HITS,
    TOX_FRAGMENT_MAX_MAP,
    TOX_FRAGMENT_MAX_PENALTY,
    TOX_FRAGMENT_MIN_SUPPORT,
    TOX_LOW_APPLICABILITY_WARNING_THRESHOLD,
    TOX_OOD_NEUTRAL_BASELINE,
)
from .rdkit_features import compute_rdkit_features
from .toxicity_classification import classify_ghs
from .utils import clamp01, validate_required_columns

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def _ensure_rdkit() -> None: return True

@dataclass(frozen = True)
class DrugToxSpace:
    feature_names: list[str]
    feature_means: np.ndarray
    feature_stds: np.ndarray
    non_toxic_centroid: np.ndarray
    toxic_centroid: np.ndarray
    distance_scale: float
    fp_non_toxic_centroid: np.ndarray
    fp_toxic_centroid: np.ndarray
    block_weights: dict[str, float]
    bit_log_odds: dict[int, float]
    bit_effect: dict[int, float]
    bit_support: dict[int, int]
    bit_fragments: dict[int, str]
    z_matrix: np.ndarray
    fp_matrix: np.ndarray
    ghs_categories: np.ndarray
    train_log_ld50: np.ndarray
    train_log_ld50_sorted: np.ndarray


def save_drug_tox_space(space: DrugToxSpace, path: Path) -> None:
    '''Persist a complete DrugToxSpace to disk as a compressed NPZ artifact.
    ---
    This format stores full k-NN state and continuous-toxicity calibration payloads'''
    
    path.parent.mkdir(parents = True, exist_ok = True)
    bit_ids = np.array(list(space.bit_log_odds.keys()), dtype = np.int32)
    bit_vals = np.array([float(space.bit_log_odds[int(k)]) for k in bit_ids], dtype = float)
    effect_ids = np.array(list(space.bit_effect.keys()), dtype = np.int32)
    effect_vals = np.array([float(space.bit_effect[int(k)]) for k in effect_ids], dtype = float)
    support_vals = np.array([int(space.bit_support.get(int(k), 0)) for k in effect_ids], dtype = np.int32)

    frag_ids = np.array(list(space.bit_fragments.keys()), dtype = np.int32)
    frag_smiles = np.array([space.bit_fragments[int(k)] for k in frag_ids], dtype = object)

    np.savez_compressed(
        path,
        feature_names = np.array(space.feature_names, dtype=object),
        feature_means = np.asarray(space.feature_means, dtype=float),
        feature_stds = np.asarray(space.feature_stds, dtype=float),
        non_toxic_centroid = np.asarray(space.non_toxic_centroid, dtype=float),
        toxic_centroid = np.asarray(space.toxic_centroid, dtype=float),
        distance_scale = np.array([float(space.distance_scale)], dtype=float),
        fp_non_toxic_centroid = np.asarray(space.fp_non_toxic_centroid, dtype=float),
        fp_toxic_centroid = np.asarray(space.fp_toxic_centroid, dtype=float),
        block_weight_numeric = np.array([float(space.block_weights.get("numeric", 0.5))], dtype=float),
        block_weight_fingerprint = np.array([float(space.block_weights.get("fingerprint", 0.5))], dtype=float),
        bit_ids = bit_ids,
        bit_log_odds = bit_vals,
        bit_effect_ids = effect_ids,
        bit_effect_vals = effect_vals,
        bit_support_vals = support_vals,
        frag_ids = frag_ids,
        frag_smiles = frag_smiles,
        z_matrix = np.asarray(space.z_matrix, dtype=float),
        fp_matrix = np.asarray(space.fp_matrix, dtype=np.int8),
        ghs_categories = np.asarray(space.ghs_categories, dtype=np.int16),
        train_log_ld50 = np.asarray(space.train_log_ld50, dtype=float),
        train_log_ld50_sorted = np.asarray(space.train_log_ld50_sorted, dtype=float),
    )


def load_drug_tox_space(path: Path) -> DrugToxSpace:
    '''
    Load a complete DrugToxSpace from an NPZ artifact.
    ---
    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required fields are missing in the artifact'''
    
    if not path.exists(): raise FileNotFoundError(f"Tox space file not found: {path}")

    required_keys = {
        "feature_names",
        "feature_means",
        "feature_stds",
        "non_toxic_centroid",
        "toxic_centroid",
        "distance_scale",
        "fp_non_toxic_centroid",
        "fp_toxic_centroid",
        "block_weight_numeric",
        "block_weight_fingerprint",
        "bit_ids",
        "bit_log_odds",
        "bit_effect_ids",
        "bit_effect_vals",
        "bit_support_vals",
        "frag_ids",
        "frag_smiles",
        "z_matrix",
        "fp_matrix",
        "ghs_categories",
        "train_log_ld50",
        "train_log_ld50_sorted",
    }
    with np.load(path, allow_pickle=True) as data:
        missing = sorted(required_keys.difference(set(data.files)))
        if missing:
            raise ValueError(
                f"Incomplete tox space artifact at {path}. Missing keys: {missing}. "
                "Rebuild using scripts/build_drug_tox_space.py with the updated code."
            )

        feature_names = [str(x) for x in data["feature_names"].tolist()]
        feature_means = np.asarray(data["feature_means"], dtype=float)
        feature_stds = np.asarray(data["feature_stds"], dtype=float)
        non_toxic_centroid = np.asarray(data["non_toxic_centroid"], dtype=float)
        toxic_centroid = np.asarray(data["toxic_centroid"], dtype=float)
        distance_scale = float(
            np.asarray(data["distance_scale"], dtype=float).reshape(-1)[0]
        )
        fp_non_toxic_centroid = np.asarray(data["fp_non_toxic_centroid"], dtype=float)
        fp_toxic_centroid = np.asarray(data["fp_toxic_centroid"], dtype=float)

        block_weights = {
            "numeric": float(
                np.asarray(data["block_weight_numeric"], dtype=float).reshape(-1)[0]
            ),
            "fingerprint": float(
                np.asarray(data["block_weight_fingerprint"], dtype=float).reshape(-1)[0]
            ),
        }

        bit_ids = np.asarray(data["bit_ids"], dtype=np.int32).tolist()
        bit_vals = np.asarray(data["bit_log_odds"], dtype=float).tolist()
        bit_log_odds = {int(k): float(v) for k, v in zip(bit_ids, bit_vals)}
        effect_ids = np.asarray(data["bit_effect_ids"], dtype=np.int32).tolist()
        effect_vals = np.asarray(data["bit_effect_vals"], dtype=float).tolist()
        support_vals = np.asarray(data["bit_support_vals"], dtype=np.int32).tolist()
        bit_effect = {int(k): float(v) for k, v in zip(effect_ids, effect_vals)}
        bit_support = {int(k): int(v) for k, v in zip(effect_ids, support_vals)}

        frag_ids = np.asarray(data["frag_ids"], dtype=np.int32).tolist()
        frag_smiles = [
            str(x) for x in np.asarray(data["frag_smiles"], dtype=object).tolist()
        ]
        bit_fragments = {int(k): str(v) for k, v in zip(frag_ids, frag_smiles)}

        z_matrix = np.asarray(data["z_matrix"], dtype=float)
        fp_matrix = np.asarray(data["fp_matrix"], dtype=np.int8)
        ghs_categories = np.asarray(data["ghs_categories"], dtype=np.int16)
        train_log_ld50 = np.asarray(data["train_log_ld50"], dtype=float)
        train_log_ld50_sorted = np.asarray(data["train_log_ld50_sorted"], dtype=float)

    return DrugToxSpace(
        feature_names = feature_names,
        feature_means = feature_means,
        feature_stds = feature_stds,
        non_toxic_centroid = non_toxic_centroid,
        toxic_centroid = toxic_centroid,
        distance_scale = distance_scale,
        fp_non_toxic_centroid = fp_non_toxic_centroid,
        fp_toxic_centroid = fp_toxic_centroid,
        block_weights = block_weights,
        bit_log_odds = bit_log_odds,
        bit_effect = bit_effect,
        bit_support = bit_support,
        bit_fragments = bit_fragments,
        z_matrix = z_matrix,
        fp_matrix = fp_matrix,
        ghs_categories = ghs_categories,
        train_log_ld50 = train_log_ld50,
        train_log_ld50_sorted = train_log_ld50_sorted)


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    z = (matrix - means) / stds
    return z, means, stds


def _tanimoto_binary_to_prob(binary: np.ndarray, prob: np.ndarray) -> float:
    '''Tanimoto similarity between a binary vector and a probability vector'''
    dot = float(np.dot(binary, prob))
    sum_a = float(binary.sum())
    sum_b = float(prob.sum())
    denom = sum_a + sum_b - dot
    if denom <= 0: return 0.0
    return dot / denom


def _fingerprint_similarity(
    binary: np.ndarray,
    centroid: np.ndarray,
) -> float:
    '''Tanimoto similarity between a binary vector and a centroid probability vector'''
    return _tanimoto_binary_to_prob(binary, centroid)


def _tanimoto_similarity_matrix(binary: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    '''Vectorized Tanimoto similarity for a binary query vs binary matrix'''
    dot = matrix @ binary
    sum_a = matrix.sum(axis = 1)
    sum_b = binary.sum()
    denom = sum_a + sum_b - dot
    return np.where(denom > 0, dot / denom, 0.0)


def _knn_numeric_similarity(
    z_query: np.ndarray,
    z_matrix: np.ndarray,
    k: int,
    distance_scale: float,
) -> tuple[float, int]:
    '''Mean similarity of k nearest numeric neighbors'''
    if z_matrix.size == 0:
        return float("nan"), 0
    distances = np.linalg.norm(z_matrix - z_query, axis=1)
    k_use = min(k, distances.shape[0])
    idx = np.argpartition(distances, k_use - 1)[:k_use]
    d = distances[idx]
    sims = 1.0 - np.clip(d / max(distance_scale, 1e-9), 0.0, 1.0)
    return float(np.mean(sims)), k_use


def _knn_fingerprint_similarity(
    fp_query: np.ndarray,
    fp_matrix: np.ndarray,
    k: int,
) -> tuple[float, int]:
    '''Mean similarity of k nearest fingerprint neighbors (by Tanimoto)'''
    if fp_matrix.size == 0: return float("nan"), 0
    sims = _tanimoto_similarity_matrix(fp_query, fp_matrix)
    k_use = min(k, sims.shape[0])
    idx = np.argpartition(-sims, k_use - 1)[:k_use]
    return float(np.mean(sims[idx])), k_use

def _fingerprint_score(
    binary: np.ndarray,
    non_toxic_centroid: np.ndarray,
    toxic_centroid: np.ndarray,
) -> float:
    '''Map fingerprint similarity to a [0,1] score (non-toxic preferred)'''
    sim_non = _tanimoto_binary_to_prob(binary, non_toxic_centroid)
    sim_tox = _tanimoto_binary_to_prob(binary, toxic_centroid)
    return clamp01((sim_non - sim_tox + 1.0) / 2.0)


def _compute_block_weights(
    numeric_scores: np.ndarray,
    fingerprint_scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    '''Compute block weights from class separation of scores'''
    tox_mask = labels.astype(bool)
    non_mask = ~tox_mask
    if tox_mask.sum() == 0 or non_mask.sum() == 0:
        return {"numeric": 0.5, "fingerprint": 0.5}

    num_effect = float(
        numeric_scores[non_mask].mean() - numeric_scores[tox_mask].mean()
    )
    fp_effect = float(
        fingerprint_scores[non_mask].mean() - fingerprint_scores[tox_mask].mean()
    )
    num_abs = abs(num_effect)
    fp_abs = abs(fp_effect)
    total = num_abs + fp_abs
    if total == 0:
        return {"numeric": 0.5, "fingerprint": 0.5}
    return {"numeric": num_abs / total, "fingerprint": fp_abs / total}


def _impute_nan(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Impute NaNs with column medians. Returns (imputed, median_values)'''
    col_median = np.nanmedian(matrix, axis = 0)
    if np.isnan(col_median).any(): raise ValueError("NaNs present in all rows for at least one feature column.")
    imputed = matrix.copy()
    nan_idx = np.where(np.isnan(imputed))
    imputed[nan_idx] = np.take(col_median, nan_idx[1])
    return imputed, col_median


def _first_non_empty_string(series: pd.Series) -> str | None:
    for value in series.tolist():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def build_pooled_acute_tox_df(
    iv_df: pd.DataFrame,
    ip_df: pd.DataFrame,
    smiles_col: str,
    ld50_col: str,
    inchikey_col: str,
) -> pd.DataFrame:
    '''
    Pool IV/IP acute toxicity datasets and deduplicate by identity.

    Dedupe key priority:
    1) InChIKey (when present)
    2) Canonical SMILES

    Target aggregation:
    - median(log10(LD50 mg/kg)) per identity key'''

    validate_required_columns(
        iv_df.columns, [smiles_col, ld50_col, inchikey_col], context="pool_iv_df"
    )
    validate_required_columns(
        ip_df.columns, [smiles_col, ld50_col, inchikey_col], context="pool_ip_df"
    )

    combined = pd.concat(
        [
            iv_df.assign(_tox_route="iv"),
            ip_df.assign(_tox_route="ip"),
        ],
        ignore_index=True,
    )
    prepared = _prepare_continuous_tox_training_df(
        combined,
        smiles_col=smiles_col,
        ld50_col=ld50_col,
        inchikey_col=inchikey_col,
        dedupe_by_identity=True,
    )
    prepared = prepared.copy()
    prepared.loc[:, "_tox_route"] = "pooled"
    return prepared


def _prepare_continuous_tox_training_df(
    df: pd.DataFrame,
    smiles_col: str,
    ld50_col: str,
    inchikey_col: str | None,
    dedupe_by_identity: bool,
) -> pd.DataFrame:
    '''Prepare a continuous-toxicity training frame with deterministic deduplication'''

    required = [smiles_col, ld50_col]
    if inchikey_col is not None:
        required.append(inchikey_col)
    validate_required_columns(
        df.columns, required, context="prepare_continuous_tox_training_df"
    )

    cols = [smiles_col, ld50_col]
    if inchikey_col is not None:
        cols.append(inchikey_col)
    work = df.loc[:, cols].copy()

    work.loc[:, smiles_col] = work[smiles_col].astype("string").str.strip()
    work.loc[:, ld50_col] = pd.to_numeric(work[ld50_col], errors="coerce")
    if inchikey_col is not None:
        work.loc[:, inchikey_col] = work[inchikey_col].astype("string").str.strip()
    else:
        inchikey_col = "InChIKey"
        work.loc[:, inchikey_col] = pd.Series([pd.NA] * len(work), dtype="string")

    before = len(work)
    valid_mask = (
        work[smiles_col].notna()
        & (work[smiles_col] != "")
        & work[ld50_col].notna()
        & (work[ld50_col] > 0)
    )
    work = work.loc[valid_mask].copy()
    dropped = before - len(work)
    if dropped > 0:
        _warnings.warn(
            f"Dropped {dropped}/{before} rows with invalid SMILES or non-positive LD50 values."
        )

    work.loc[:, "log10_ld50_mgkg"] = np.log10(work[ld50_col].astype(float))

    inchikey_present = work[inchikey_col].notna() & (work[inchikey_col] != "")
    work.loc[:, "dedupe_key"] = np.where(
        inchikey_present,
        "ik:" + work[inchikey_col].astype(str),
        "smi:" + work[smiles_col].astype(str),
    )

    if not dedupe_by_identity:
        out = work.copy()
        out.loc[:, "records_merged"] = 1
        return out

    grouped = (
        work.groupby("dedupe_key", sort=True, as_index=False)
        .agg(
            {
                smiles_col: _first_non_empty_string,
                inchikey_col: _first_non_empty_string,
                "log10_ld50_mgkg": "median",
            }
        )
        .rename(columns={"log10_ld50_mgkg": "_median_log10_ld50_mgkg"})
    )
    counts = work.groupby("dedupe_key", sort=True).size().rename("records_merged")
    out = grouped.merge(counts, on="dedupe_key", how="left", validate="one_to_one")
    out.loc[:, "log10_ld50_mgkg"] = out["_median_log10_ld50_mgkg"].astype(float)
    out.loc[:, ld50_col] = np.power(
        10.0, out["log10_ld50_mgkg"]
    )  # back-transform for legacy GHS usage
    out = out.drop(columns=["_median_log10_ld50_mgkg"])
    return out


def build_tox_feature_matrix(
    df: pd.DataFrame,
    smiles_col: str,
    feature_names: Iterable[str] | None = None,
    include_3d: bool | None = None,
    show_progress: bool = False,
    skip_invalid: bool = False,
    return_stats: bool = False,
) -> (
    tuple[np.ndarray, list[str], list[list[str]]]
    | tuple[np.ndarray, list[str], list[list[str]], dict[str, object]]
):
    '''Build a numeric feature matrix from SMILES using RDKit descriptors.
    Returns (matrix, feature_names, warnings_per_row).

    skip_invalid skips SMILES that fail RDKit parsing.
    return_stats returns a dict with skip/keep counts and example failures'''

    feature_names = list(feature_names or TOX_SPACE_NUMERIC_FEATURES)
    validate_required_columns(
        df.columns, [smiles_col], context="build_tox_feature_matrix"
    )

    # Enable 3D when DipoleMoment is requested
    if include_3d is None:
        include_3d = "DipoleMoment" in feature_names

    rows: list[list[float]] = []
    warnings_per_row: list[list[str]] = []
    kept_indices: list[int] = []
    skipped: list[dict[str, str | int]] = []

    smiles_list = df[smiles_col].tolist()
    iterator: Iterable[tuple[int, str]] = enumerate(smiles_list)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(
            list(enumerate(smiles_list)), desc="Tox descriptors", leave=False
        )

    for idx, smi in iterator:
        try:
            feats, warn = compute_rdkit_features(
                smi, include_fingerprints=False, include_3d=include_3d
            )
        except Exception as e:
            if not skip_invalid:
                raise ValueError(
                    f"Failed to featurize SMILES at row {idx}: {smi}"
                ) from e
            skipped.append(
                {"row": idx, "smiles": str(smi), "error": f"{type(e).__name__}: {e}"}
            )
            continue
        rows.append([float(feats[f]) for f in feature_names])
        warnings_per_row.append(warn)
        kept_indices.append(idx)

    if not rows:
        raise ValueError("No valid SMILES available to build tox feature matrix.")

    if skip_invalid and skipped:
        total = len(smiles_list)
        skipped_count = len(skipped)
        skipped_pct = (skipped_count / total * 100.0) if total else 0.0
        examples = "; ".join(
            f"row {ex['row']} smiles={ex['smiles']} err={ex['error']}"
            for ex in skipped[:3]
        )
        _warnings.warn(
            f"Skipped {skipped_count}/{total} SMILES ({skipped_pct:.2f}%) during tox feature extraction. "
            f"Examples: {examples}"
        )

    matrix = np.array(rows, dtype=float)
    non_finite_mask = ~np.isfinite(matrix)
    if non_finite_mask.any():
        n_non_finite = int(non_finite_mask.sum())
        matrix = matrix.copy()
        matrix[non_finite_mask] = np.nan
        _warnings.warn(
            f"Converted {n_non_finite} non-finite descriptor values (NaN/Inf) to NaN before imputation."
        )

    if np.isnan(matrix).any():
        # Drop columns that are entirely NaN to keep the tox space buildable.
        all_nan_cols = np.isnan(matrix).all(axis=0)
        if all_nan_cols.any():
            dropped = [feature_names[i] for i, flag in enumerate(all_nan_cols) if flag]
            matrix = matrix[:, ~all_nan_cols]
            feature_names = [
                f for i, f in enumerate(feature_names) if not all_nan_cols[i]
            ]
            for warn in warnings_per_row:
                warn.append(f"Dropped all-NaN descriptors: {dropped}")

        if np.isnan(matrix).any():
            matrix, _ = _impute_nan(matrix)

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
        return matrix, feature_names, warnings_per_row, stats

    return matrix, feature_names, warnings_per_row


def _compute_bit_log_odds(
    fps: np.ndarray, labels: np.ndarray, alpha: float = 1.0
) -> dict[int, float]:
    '''Compute per-bit log-odds ratio between toxic and non-toxic classes'''

    tox_mask = labels.astype(bool)
    non_mask = ~tox_mask
    tox_count = tox_mask.sum()
    non_count = non_mask.sum()
    if tox_count == 0 or non_count == 0:
        raise ValueError("Both toxic and non-toxic samples are required for log-odds.")

    tox_bits = fps[tox_mask].sum(axis=0)
    non_bits = fps[non_mask].sum(axis=0)

    bit_log_odds: dict[int, float] = {}
    for idx in range(fps.shape[1]):
        p_tox = (tox_bits[idx] + alpha) / (tox_count + 2 * alpha)
        p_non = (non_bits[idx] + alpha) / (non_count + 2 * alpha)
        log_odds = float(np.log(p_tox / (1.0 - p_tox)) - np.log(p_non / (1.0 - p_non)))
        bit_log_odds[idx] = log_odds
    return bit_log_odds


def _compute_bit_effects(
    fps: np.ndarray,
    log_ld50: np.ndarray,
    min_support: int,
) -> tuple[dict[int, float], dict[int, int]]:
    '''Compute per-bit continuous toxicity effect:
    bit_effect = mean(logLD50 | bit=1) - mean(logLD50 | bit=0)

    Negative effect => bit is associated with lower LD50 (more toxic)'''

    if fps.shape[0] != log_ld50.shape[0]:
        raise ValueError("fps/log_ld50 row mismatch in _compute_bit_effects")

    total_y = float(np.sum(log_ld50))
    bit_effect: dict[int, float] = {}
    bit_support: dict[int, int] = {}

    for bit_id in range(fps.shape[1]):
        bit_col = fps[:, bit_id].astype(bool)
        support = int(bit_col.sum())
        if support < min_support:
            continue
        off = int(fps.shape[0] - support)
        if off <= 0:
            continue

        y_on = float(np.sum(log_ld50[bit_col]))
        y_off = total_y - y_on
        mean_on = y_on / support
        mean_off = y_off / off
        effect = float(mean_on - mean_off)

        bit_effect[bit_id] = effect
        bit_support[bit_id] = support

    return bit_effect, bit_support


def _bit_to_fragment_smiles(
    mol: "Chem.Mol", bit_id: int, bit_info: dict[int, list[tuple[int, int]]]
) -> str | None:
    '''Extract a representative fragment SMILES for a given Morgan bit'''

    _ensure_rdkit()
    if bit_id not in bit_info:
        return None
    atom_idx, radius = bit_info[bit_id][0]
    if hasattr(rdMolDescriptors, "FindAtomEnvironmentOfRadiusN"):
        env = rdMolDescriptors.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    else:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    if not env:
        return None
    submol = Chem.PathToSubmol(mol, env)
    return Chem.MolToSmiles(submol)


def _extract_fragment_map(
    smiles_list: list[str],
    bit_ids: Iterable[int],
) -> dict[int, str]:
    '''Map Morgan fingerprint bits to a representative fragment SMILES'''

    _ensure_rdkit()
    bit_ids = set(bit_ids)
    bit_fragments: dict[int, str] = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bit_info: dict[int, list[tuple[int, int]]] = {}
        rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048, bitInfo=bit_info
        )
        for bit_id in list(bit_ids):
            frag = _bit_to_fragment_smiles(mol, bit_id, bit_info)
            if frag:
                bit_fragments[bit_id] = frag
                bit_ids.remove(bit_id)
        if not bit_ids:
            break
    return bit_fragments


def build_drug_tox_space(
    df: pd.DataFrame,
    smiles_col: str,
    ld50_col: str,
    feature_names: Iterable[str] | None = None,
    max_fragments: int = TOX_FRAGMENT_MAX_MAP,
    show_progress: bool = False,
    skip_invalid: bool = False,
    return_stats: bool = False,
    inchikey_col: str | None = None,
    dedupe_by_identity: bool = True,
    min_fragment_support: int = TOX_FRAGMENT_MIN_SUPPORT,
) -> DrugToxSpace | tuple[DrugToxSpace, dict[str, object]]:
    '''Build a toxicity feature space with continuous LD50 and fragment effect modeling.
    ---
    Key behaviors:
    - Continuous target: log10(LD50 mg/kg)
    - Deterministic dedupe by InChIKey (fallback Canonical SMILES)
    - Mixed numeric/fingerprint kNN similarity payload
    - Bit-level fragment effects for toxicity feedback'''

    _ensure_rdkit()
    prepared = _prepare_continuous_tox_training_df(
        df,
        smiles_col=smiles_col,
        ld50_col=ld50_col,
        inchikey_col=inchikey_col,
        dedupe_by_identity=dedupe_by_identity,
    )
    classified = classify_ghs(prepared, ld50_col=ld50_col)

    stats: dict[str, object] | None = None
    if skip_invalid:
        matrix, feat_names, _, stats = build_tox_feature_matrix(
            classified,
            smiles_col,
            feature_names,
            show_progress=show_progress,
            skip_invalid=True,
            return_stats=True,
        )
        kept_indices = stats["kept_indices"] if stats else []
        classified = classified.iloc[kept_indices].reset_index(drop=True)
    else:
        matrix, feat_names, _ = build_tox_feature_matrix(
            classified,
            smiles_col,
            feature_names,
            show_progress=show_progress,
        )

    labels = classified["ghs_toxic"].astype(bool).to_numpy()
    z, means, stds = _standardize(matrix)

    tox_mask = labels.astype(bool)
    non_mask = ~tox_mask
    if non_mask.sum() > 0:
        non_centroid = z[non_mask].mean(axis=0)
    else:
        non_centroid = z.mean(axis=0)
        _warnings.warn(
            "No non-toxic samples available; non-toxic centroid fallback uses global mean."
        )

    if tox_mask.sum() > 0:
        tox_centroid = z[tox_mask].mean(axis=0)
    else:
        tox_centroid = z.mean(axis=0)
        _warnings.warn(
            "No toxic samples available; toxic centroid fallback uses global mean."
        )

    # Distance scale for numeric similarity
    distances = np.linalg.norm(z - non_centroid, axis=1)
    distance_scale = float(np.quantile(distances, 0.75))
    if distance_scale <= 0:
        distance_scale = float(np.max(distances)) if np.max(distances) > 0 else 1.0

    # Fingerprints for fragment analysis
    fps = []
    smiles_list = classified[smiles_col].astype(str).tolist()
    iterator: Iterable[tuple[int, str]] = enumerate(smiles_list)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(
            list(enumerate(smiles_list)), desc="Tox fingerprints", leave=False
        )

    for idx, smi in iterator:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(
                f"Invalid SMILES at row {idx} while building fingerprints: {smi}"
            )
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps.append(np.array([int(x) for x in fp.ToBitString()], dtype=int))
    fps_matrix = np.vstack(fps)

    if non_mask.sum() > 0:
        fp_non_centroid = fps_matrix[non_mask].mean(axis=0)
    else:
        fp_non_centroid = fps_matrix.mean(axis=0)

    if tox_mask.sum() > 0:
        fp_tox_centroid = fps_matrix[tox_mask].mean(axis=0)
    else:
        fp_tox_centroid = fps_matrix.mean(axis=0)

    train_log_ld50 = classified["log10_ld50_mgkg"].astype(float).to_numpy()
    train_log_ld50_sorted = np.sort(train_log_ld50)

    if tox_mask.sum() > 0 and non_mask.sum() > 0:
        bit_log_odds = _compute_bit_log_odds(fps_matrix, labels)
        numeric_scores = 1.0 - np.clip(distances / max(distance_scale, 1e-9), 0.0, 1.0)
        fp_scores = np.array(
            [
                _fingerprint_score(fps_matrix[i], fp_non_centroid, fp_tox_centroid)
                for i in range(fps_matrix.shape[0])
            ],
            dtype=float,
        )
        block_weights = _compute_block_weights(numeric_scores, fp_scores, labels)
    else:
        bit_log_odds = {bit_id: 0.0 for bit_id in range(fps_matrix.shape[1])}
        block_weights = {"numeric": 0.5, "fingerprint": 0.5}

    bit_effect, bit_support = _compute_bit_effects(
        fps_matrix,
        train_log_ld50,
        min_support=min_fragment_support,
    )

    if bit_effect:
        top_bits = sorted(bit_effect, key=lambda k: abs(bit_effect[k]), reverse=True)[
            :max_fragments
        ]
    else:
        top_bits = sorted(bit_log_odds, key=lambda k: abs(bit_log_odds[k]), reverse=True)[
            :max_fragments
        ]
    bit_fragments = _extract_fragment_map(smiles_list, top_bits)

    space = DrugToxSpace(
        feature_names=list(feat_names),
        feature_means=means,
        feature_stds=stds,
        non_toxic_centroid=non_centroid,
        toxic_centroid=tox_centroid,
        distance_scale=distance_scale,
        fp_non_toxic_centroid=fp_non_centroid,
        fp_toxic_centroid=fp_tox_centroid,
        block_weights=block_weights,
        bit_log_odds=bit_log_odds,
        bit_effect=bit_effect,
        bit_support=bit_support,
        bit_fragments=bit_fragments,
        z_matrix=z,
        fp_matrix=fps_matrix.astype(np.int8),
        ghs_categories=classified["ghs_category"].fillna(0).astype(int).to_numpy(),
        train_log_ld50=train_log_ld50,
        train_log_ld50_sorted=train_log_ld50_sorted,
    )

    if return_stats:
        if stats is None:
            total = len(df)
            stats = {
                "total": total,
                "kept": len(matrix),
                "skipped": total - len(matrix),
                "skipped_pct": (
                    ((total - len(matrix)) / total * 100.0) if total else 0.0
                ),
                "kept_indices": list(range(len(matrix))),
                "skipped_examples": [],
            }
        return space, stats

    return space


def tox_proximity_score(
    space: DrugToxSpace,
    feature_vector: np.ndarray,
    fingerprint_bits: np.ndarray | list[int] | None = None,
) -> float:
    '''Convert distance to non-toxic centroid into a [0,1] proximity score'''

    z = (feature_vector - space.feature_means) / space.feature_stds
    dist = float(np.linalg.norm(z - space.non_toxic_centroid))
    numeric_score = 1.0 - clamp01(dist / max(space.distance_scale, 1e-9))

    if fingerprint_bits is None:
        return numeric_score

    fp_bits = np.array(fingerprint_bits, dtype=float)
    fp_score = _fingerprint_score(
        fp_bits, space.fp_non_toxic_centroid, space.fp_toxic_centroid
    )
    weights = space.block_weights or {"numeric": 1.0, "fingerprint": 0.0}
    return clamp01(
        weights["numeric"] * numeric_score + weights["fingerprint"] * fp_score
    )


def tox_proximity_score_toxic(
    space: DrugToxSpace,
    feature_vector: np.ndarray,
    fingerprint_bits: np.ndarray | list[int] | None = None,
) -> float:
    '''Convert distance to toxic centroid into a [0,1] proximity score.
    Higher means more similar to toxic space'''

    z = (feature_vector - space.feature_means) / space.feature_stds
    dist = float(np.linalg.norm(z - space.toxic_centroid))
    numeric_score = 1.0 - clamp01(dist / max(space.distance_scale, 1e-9))

    if fingerprint_bits is None:
        return numeric_score

    fp_bits = np.array(fingerprint_bits, dtype=float)
    fp_score = _fingerprint_similarity(fp_bits, space.fp_toxic_centroid)
    weights = space.block_weights or {"numeric": 1.0, "fingerprint": 0.0}
    return clamp01(
        weights["numeric"] * numeric_score + weights["fingerprint"] * fp_score
    )


def tox_proximity_by_ghs_category(
    space: DrugToxSpace,
    feature_vector: np.ndarray,
    fingerprint_bits: np.ndarray | list[int],
    k: int,
    severity_alpha: float,
) -> tuple[dict[int, float], float, list[str]]:
    '''Compute per-category proximity using k-NN and return severity-weighted toxic proximity.
    Category 5 is treated as non-toxic and excluded from the toxic aggregation'''

    z_query = (feature_vector - space.feature_means) / space.feature_stds
    fp_query = np.array(fingerprint_bits, dtype=float)
    weights = space.block_weights or {"numeric": 1.0, "fingerprint": 0.0}

    cat_scores: dict[int, float] = {}
    warnings: list[str] = []

    for cat in [1, 2, 3, 4, 5]:
        idx = np.where(space.ghs_categories == cat)[0]
        if idx.size == 0:
            cat_scores[cat] = float("nan")
            warnings.append(f"GHS cat {cat}: no samples available for k-NN proximity")
            continue

        z_cat = space.z_matrix[idx]
        fp_cat = space.fp_matrix[idx]
        num_sim, num_k = _knn_numeric_similarity(
            z_query, z_cat, k, space.distance_scale
        )
        fp_sim, fp_k = _knn_fingerprint_similarity(fp_query, fp_cat, k)
        if idx.size < k:
            warnings.append(
                f"GHS cat {cat}: k-NN used {idx.size}/{k} samples (category has {idx.size})"
            )

        combined = weights["numeric"] * num_sim + weights["fingerprint"] * fp_sim
        cat_scores[cat] = clamp01(combined)

    # Severity-weighted toxic proximity (cats 1-4 only)
    weighted_sum = 0.0
    weight_total = 0.0
    for cat in sorted(GHS_TOXIC_CATEGORIES):
        score = cat_scores.get(cat)
        if score is None or np.isnan(score):
            continue
        weight = float(np.exp(severity_alpha * (5 - cat)))
        weighted_sum += weight * score
        weight_total += weight
    tox_prox_toxic = weighted_sum / weight_total if weight_total > 0 else float("nan")
    return cat_scores, clamp01(tox_prox_toxic), warnings


def tox_ld50_safety_profile(
    space: DrugToxSpace,
    feature_vector: np.ndarray,
    fingerprint_bits: np.ndarray | list[int],
    *,
    k: int,
    low_applicability_warning_threshold: float = TOX_LOW_APPLICABILITY_WARNING_THRESHOLD,
    neutral_baseline: float = TOX_OOD_NEUTRAL_BASELINE,
    max_fragment_penalty: float = TOX_FRAGMENT_MAX_PENALTY,
    max_fragment_hits: int = TOX_FRAGMENT_MAX_HITS,
) -> tuple[dict[str, object], list[str]]:
    '''
    Continuous-LD50 toxicity profile with applicability gating and fragment penalties.
    ---
    Returns a dict with scoring-ready values and explainability payload'''

    z_query = (feature_vector - space.feature_means) / space.feature_stds
    fp_query = np.array(fingerprint_bits, dtype=float)

    weights = space.block_weights or {"numeric": 1.0, "fingerprint": 0.0}

    num_dist = np.linalg.norm(space.z_matrix - z_query, axis=1)
    num_sim = 1.0 - np.clip(num_dist / max(space.distance_scale, 1e-9), 0.0, 1.0)
    fp_sim = _tanimoto_similarity_matrix(fp_query, space.fp_matrix)
    combined_sim = np.clip(
        weights["numeric"] * num_sim + weights["fingerprint"] * fp_sim,
        0.0,
        1.0,
    )

    k_use = min(max(int(k), 1), combined_sim.shape[0])
    top_idx = np.argsort(-combined_sim, kind="mergesort")[:k_use]
    top_w = combined_sim[top_idx]

    applicability = float(np.mean(top_w))
    w_sum = float(np.sum(top_w))
    if w_sum <= 1e-12:
        pred_log10 = float(np.mean(space.train_log_ld50))
    else:
        pred_log10 = float(np.sum(space.train_log_ld50[top_idx] * top_w) / w_sum)

    sorted_ld50 = space.train_log_ld50_sorted
    rank = int(np.searchsorted(sorted_ld50, pred_log10, side="right"))
    tox_safety_raw_ld50 = clamp01(rank / max(len(sorted_ld50), 1))

    tox_safety_cal = clamp01(
        applicability * tox_safety_raw_ld50 + (1.0 - applicability) * neutral_baseline
    )

    toxic_bit_weights = {
        bit_id: abs(effect)
        for bit_id, effect in space.bit_effect.items()
        if effect < 0
    }
    toxic_weight_total = float(sum(toxic_bit_weights.values()))

    active_bits = np.flatnonzero(fp_query > 0.5).astype(int).tolist()
    active_toxic_hits: list[dict[str, object]] = []
    active_toxic_weight = 0.0
    for bit_id in active_bits:
        if bit_id not in toxic_bit_weights:
            continue
        effect = float(space.bit_effect[bit_id])
        weight = float(abs(effect))
        active_toxic_weight += weight
        active_toxic_hits.append(
            {
                "bit_id": int(bit_id),
                "effect": effect,
                "support": int(space.bit_support.get(bit_id, 0)),
                "fragment_smiles": str(space.bit_fragments.get(bit_id, "")),
            }
        )

    active_toxic_hits = sorted(
        active_toxic_hits,
        key=lambda item: abs(float(item["effect"])),
        reverse=True,
    )
    fragment_hit_count = len(active_toxic_hits)
    fragment_hits = active_toxic_hits[:max_fragment_hits]

    if toxic_weight_total <= 1e-12:
        fragment_risk = 0.0
    else:
        fragment_risk = clamp01(active_toxic_weight / toxic_weight_total)

    fragment_penalty = min(
        max_fragment_penalty,
        max_fragment_penalty * fragment_risk * applicability,
    )

    toxicity_term = clamp01(tox_safety_cal - fragment_penalty)

    warnings: list[str] = []
    if applicability < low_applicability_warning_threshold:
        warnings.append(
            f"Toxicity applicability is low ({applicability:.3f} < {low_applicability_warning_threshold:.3f}); "
            f"toxicity safety was neutralized toward {neutral_baseline:.2f}."
        )

    payload: dict[str, object] = {
        "tox_ld50_pred_log10": pred_log10,
        "tox_ld50_pred_mgkg": float(np.power(10.0, pred_log10)),
        "tox_safety_raw_ld50": tox_safety_raw_ld50,
        "tox_applicability": applicability,
        "tox_safety_cal": tox_safety_cal,
        "fragment_risk": fragment_risk,
        "fragment_penalty": fragment_penalty,
        "fragment_hit_count": fragment_hit_count,
        "fragment_hits": fragment_hits,
        "toxicity_term": toxicity_term,
    }
    return payload, warnings
