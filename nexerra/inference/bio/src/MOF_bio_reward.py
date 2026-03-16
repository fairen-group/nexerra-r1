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
#  Component design (all scaled to [0,1]):
#  1) `safety`: metal safety, severity-weighted toxicity-space proximity, formal charge
#  2) `performance`: linker length, solubility, surface/loading proxies
#  3) `benign_linkers`: charge + payload-interaction proxy + PAINS/BRENK/NIH burden
#  4) `stability`: metal/linker stability, solubility, charge-release stability
#  Please note: Parts of this code are still under-development [...]
#  MIT License
#  Copyright (c) 2026 The Authors
# ---------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm as _tqdm

from .constants import (
    ACUTE_TOX_COLS,
    BENIGN_LINKER_SUBWEIGHTS,
    DRUG_TOX_SPACE_FULL_PATH,
    LINKER_STABILITY_SUBWEIGHTS,
    MOF_PROPERTIES_COLS,
    PERFORMANCE_SUBWEIGHTS,
    REWARD_BLOCK_WEIGHTS,
    REWARD_SHOW_PROGRESS,
    REWARD_SKIP_INVALID_SMILES,
    REWARD_CONTEXT_STATS_PATH,
    REWARD_FORCE_RECALCULATE_CONTEXT_STATS,
    REWARD_FIXED_LINKER_STATS,
    REWARD_FIXED_MOF_STATS,
    REWARD_LOGP_WINDOW,
    REWARD_SOLUBILITY_LOGS_RANGE,
    REWARD_TPSA_WINDOW,
    REWARD_USE_FIXED_CONTEXT_STATS,
    SAFETY_SUBWEIGHTS,
    STABILITY_SUBWEIGHTS,
    PENALTY_WEIGHTS,
    RDKIT_DEFAULT_INCLUDE_3D,
    RDKIT_DEFAULT_INCLUDE_FINGERPRINTS,
    TOX_KNN_K,
    TOX_SEVERITY_EXP_ALPHA,
)

from .data_loading import (
    load_acute_tox_ip,
    load_acute_tox_iv,
    load_metal_toxicity,
    load_mof_properties,
)

from .drug_tox_space import (
    DrugToxSpace,
    build_pooled_acute_tox_df,
    build_drug_tox_space,
    load_drug_tox_space,
    tox_ld50_safety_profile,
    tox_proximity_by_ghs_category,
    tox_proximity_score,
)

from .metal_toxicity import MetalToxicityEntry, build_metal_toxicity_table, metal_safety_score

from .rdkit_features import (
    anchor_env_symmetry_proxy,
    anchor_to_anchor_graph_distance,
    compute_ecfp4_bits,
    compute_formal_charge,
    compute_rdkit_features,
    graph_diameter,
    mol_from_smiles,
    rdkit_alert_counts_from_smiles,
    topological_symmetry_proxy,
)

from .utils import (
    clamp01,
    normalize_minmax,
    strip_lr_smiles,
    validate_required_columns,
    zlike_window,
)

# ----- Key input variables -----
DEFAULT_TOX_SPACE_PATH: Path = DRUG_TOX_SPACE_FULL_PATH

@dataclass(frozen = True)
class RewardConfig:
    component_weights: dict[str, float]
    safety_subweights: dict[str, float]
    performance_subweights: dict[str, float]
    benign_subweights: dict[str, float]
    stability_subweights: dict[str, float]
    linker_stability_subweights: dict[str, float]
    penalty_weights: dict[str, float]
    solubility_logS_range: tuple[float, float]
    logp_window: tuple[float, float]
    tpsa_window: tuple[float, float]

@dataclass(frozen = True)
class RewardContext:
    metal_toxicity: dict[str, MetalToxicityEntry]
    mof_stats: dict[str, tuple[float, float]]
    linker_stats: dict[str, tuple[float, float]]
    tox_space: DrugToxSpace | None
    stats_source: str

@dataclass(frozen = True)
class RewardResult:
    total_score: float
    components: dict[str, float]
    features: dict[str, object]
    warnings: list[str]

def _default_config() -> RewardConfig:
    return RewardConfig(
        component_weights = REWARD_BLOCK_WEIGHTS,
        safety_subweights = SAFETY_SUBWEIGHTS,
        performance_subweights = PERFORMANCE_SUBWEIGHTS,
        benign_subweights = BENIGN_LINKER_SUBWEIGHTS,
        stability_subweights = STABILITY_SUBWEIGHTS,
        linker_stability_subweights = LINKER_STABILITY_SUBWEIGHTS,
        penalty_weights = PENALTY_WEIGHTS,
        solubility_logS_range = REWARD_SOLUBILITY_LOGS_RANGE,
        logp_window = REWARD_LOGP_WINDOW,
        tpsa_window = REWARD_TPSA_WINDOW,
    )

def _compute_linker_length_metrics(raw_smiles: str, cleaned_smiles: str) -> dict[str, float]:
    '''
    Compute linker length proxies.
    ---
    Priority:
    1) Anchor-to-anchor graph distance on raw branch SMILES with [Lr]
    2) Graph diameter fallback on cleaned linker SMILES
    '''

    # Use only the minimal call-path needed for linker length proxies.
    mol = mol_from_smiles(cleaned_smiles)
    heavy_atoms = float(mol.GetNumHeavyAtoms())
    anchor_distance = anchor_to_anchor_graph_distance(raw_smiles)
    if math.isnan(anchor_distance): anchor_distance = graph_diameter(cleaned_smiles)
    return {
        "heavy_atom_count": heavy_atoms,
        "anchor_distance": float(anchor_distance),
    }


def _fit_linker_stats(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    '''Fit min/max stats for linker length proxies across the dataset'''
    smiles_col = MOF_PROPERTIES_COLS["branch_smiles"]
    validate_required_columns(df.columns, [smiles_col], context = "fit_linker_stats")
    heavy_vals: list[float] = []
    anchor_distance_vals: list[float] = []
    invalid_rows: list[tuple[int, str, str, str]] = []
    for idx, smi in enumerate(df[smiles_col].astype(str).tolist()):
        cleaned = strip_lr_smiles(smi)
        try: feats = _compute_linker_length_metrics(smi, cleaned)
        except Exception as e:
            invalid_rows.append((idx, smi, cleaned, f"{type(e).__name__}: {e}"))
            continue
        heavy_vals.append(feats["heavy_atom_count"])
        anchor_distance_vals.append(feats["anchor_distance"])
    if not heavy_vals or not anchor_distance_vals: raise ValueError("No valid linker SMILES available to fit linker stats.")
    if invalid_rows:
        import warnings as _warnings
        total = len(df)
        skipped = len(invalid_rows)
        pct = (skipped / total * 100.0) if total else 0.0
        sample = "; ".join(
            [f"row {i} cleaned={c[:60]}... err={err}" for i, _, c, err in invalid_rows[:5]]
        )
        _warnings.warn(
            f"Skipped {skipped}/{total} linker rows ({pct:.2f}%) that failed RDKit parsing. "
            f"Examples: {sample}",
            RuntimeWarning,
        )
    return {
        "heavy_atom_count": (min(heavy_vals), max(heavy_vals)),
        "anchor_distance": (min(anchor_distance_vals), max(anchor_distance_vals)),
    }


def _fit_mof_stats(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    '''Fit min/max stats for optional MOF-level properties (agsa, pld, lcd)'''
    stats: dict[str, tuple[float, float]] = {}
    for col_key in ["agsa", "pld", "lcd"]:
        col = MOF_PROPERTIES_COLS[col_key]
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        stats[col_key] = (float(values.min()), float(values.max()))
    return stats


def _validate_stats_dict(
    stats: dict[str, tuple[float, float]],
    required_keys: list[str],
    context: str,
) -> None:
    missing = [k for k in required_keys if k not in stats]
    if missing: raise ValueError(f"{context} is missing required keys: {missing}")
    for key in required_keys:
        min_val, max_val = stats[key]
        if not (isinstance(min_val, (int, float)) and isinstance(max_val, (int, float))):
            raise ValueError(f"{context}[{key}] must be numeric min/max.")
        if float(max_val) <= float(min_val):
            raise ValueError(
                f"{context}[{key}] has invalid range: min={min_val}, max={max_val}."
            )


def _save_context_stats_artifact(
    mof_stats: dict[str, tuple[float, float]],
    linker_stats: dict[str, tuple[float, float]],
    stats_source: str,
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stats_source": stats_source,
        "mof_stats": {k: [float(v[0]), float(v[1])] for k, v in mof_stats.items()},
        "linker_stats": {k: [float(v[0]), float(v[1])] for k, v in linker_stats.items()},
    }
    REWARD_CONTEXT_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REWARD_CONTEXT_STATS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

def build_reward_context(
    acute_tox_df: pd.DataFrame | None = None,
    mof_properties_df: pd.DataFrame | None = None,
    tox_space: DrugToxSpace | None = None,
) -> RewardContext:
    '''
    Build reward context:
    - metal toxicity table
    - MOF property normalization stats
    - linker length normalization stats
    - toxicity feature space (can be injected for speed)'''

    metal_df = load_metal_toxicity()
    metal_table = build_metal_toxicity_table(metal_df)
    use_fixed_stats = (
        REWARD_USE_FIXED_CONTEXT_STATS and not REWARD_FORCE_RECALCULATE_CONTEXT_STATS
    )

    if use_fixed_stats:
        mof_stats = {
            key: (float(bounds[0]), float(bounds[1]))
            for key, bounds in REWARD_FIXED_MOF_STATS.items()
        }
        linker_stats = {
            key: (float(bounds[0]), float(bounds[1]))
            for key, bounds in REWARD_FIXED_LINKER_STATS.items()
        }
        if mof_stats:
            _validate_stats_dict(
                mof_stats,
                required_keys=list(mof_stats.keys()),
                context="REWARD_FIXED_MOF_STATS",
            )
        _validate_stats_dict(
            linker_stats,
            required_keys=["heavy_atom_count", "anchor_distance"],
            context="REWARD_FIXED_LINKER_STATS",
        )
        stats_source = "fixed_constants"
    else:
        if mof_properties_df is None:
            mof_properties_df = load_mof_properties()
        mof_stats = _fit_mof_stats(mof_properties_df)
        linker_stats = _fit_linker_stats(mof_properties_df)
        stats_source = "dataset_fit"

    if tox_space is None:
        if acute_tox_df is None:
            acute_tox_df = build_pooled_acute_tox_df(
                load_acute_tox_iv(),
                load_acute_tox_ip(),
                smiles_col=ACUTE_TOX_COLS["canonical_smiles"],
                ld50_col=ACUTE_TOX_COLS["tox_value"],
                inchikey_col=ACUTE_TOX_COLS["inchikey"],
            )
        tox_space = build_drug_tox_space(
            acute_tox_df,
            smiles_col=ACUTE_TOX_COLS["canonical_smiles"],
            ld50_col=ACUTE_TOX_COLS["tox_value"],
            inchikey_col=ACUTE_TOX_COLS["inchikey"],
            dedupe_by_identity=True,
        )

    _save_context_stats_artifact(mof_stats, linker_stats, stats_source)

    return RewardContext(
        metal_toxicity=metal_table,
        mof_stats=mof_stats,
        linker_stats=linker_stats,
        tox_space=tox_space,
        stats_source=stats_source,
    )


def load_reusable_reward_context(
    tox_space_path: Path | None = None,
    mof_properties_df: pd.DataFrame | None = None,
) -> RewardContext:
    '''
    Build reward context using a precomputed tox-space artifact.
    ---
    This is the preferred path for iterative scoring loops (generator feedback),
    because tox-space fitting is skipped and category k-NN remains available'''

    path = tox_space_path or DEFAULT_TOX_SPACE_PATH
    try:
        tox_space = load_drug_tox_space(path)
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(
            f"Failed to load reusable tox space from {path}. Build it first with scripts/build_drug_tox_space.py"
        ) from e
    return build_reward_context(mof_properties_df=mof_properties_df, tox_space=tox_space)


def _score_length(
    raw_smiles: str,
    cleaned_smiles: str,
    linker_stats: dict[str, tuple[float, float]],
) -> float:
    '''Score linker length using anchor distance and heavy-atom span proxies'''
    feats = _compute_linker_length_metrics(raw_smiles, cleaned_smiles)
    heavy_min, heavy_max = linker_stats["heavy_atom_count"]
    dist_min, dist_max = linker_stats["anchor_distance"]
    heavy_score = normalize_minmax(feats["heavy_atom_count"], heavy_min, heavy_max)
    dist_score = normalize_minmax(feats["anchor_distance"], dist_min, dist_max)
    return 0.35 * heavy_score + 0.65 * dist_score


def _score_solubility(logs: float, logp: float, config: RewardConfig) -> float:
    logS_min, logS_max = config.solubility_logS_range
    logS_score = normalize_minmax(logs, logS_min, logS_max)
    logp_score = zlike_window(logp, config.logp_window[0], config.logp_window[1])
    return 0.7 * logS_score + 0.3 * logp_score


def _score_formal_charge_safety(formal_charge: int) -> float:
    '''Safety-oriented charge score (higher is safer)'''
    abs_charge = abs(formal_charge)
    if abs_charge <= 1: return 1.0
    if abs_charge == 2: return 0.6
    return 0.2


def _score_payload_interaction(logp: float, tpsa: float, config: RewardConfig) -> float:
    '''
    Low payload interaction risk proxy:
    - moderate logP window
    - moderate TPSA window'''

    logp_term = zlike_window(logp, config.logp_window[0], config.logp_window[1])
    tpsa_term = zlike_window(tpsa, config.tpsa_window[0], config.tpsa_window[1])
    return 0.5 * logp_term + 0.5 * tpsa_term


def _score_surface_area(
    agsa: float | None,
    linker_feats: dict[str, float | int | list[int]],
    mof_stats: dict[str, tuple[float, float]],
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    
    if agsa is not None and not (isinstance(agsa, float) and math.isnan(agsa)) and "agsa" in mof_stats:
        min_val, max_val = mof_stats["agsa"]
        return normalize_minmax(agsa, min_val, max_val), warnings
    # Proxy: use TPSA as a surface-area-like signal when MOF ASA is missing
    tpsa = float(linker_feats["TPSA"])
    warnings.append("Surface area: agsa missing; using TPSA proxy")
    return clamp01(tpsa / 200.0), warnings

def _score_pld(
    pld: float | None,
    linker_length_score: float,
    mof_stats: dict[str, tuple[float, float]],
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    if pld is not None and not (isinstance(pld, float) and math.isnan(pld)) and "pld" in mof_stats:
        min_val, max_val = mof_stats["pld"]
        return normalize_minmax(pld, min_val, max_val), warnings
    warnings.append("PLD: missing; using linker length proxy")
    return linker_length_score, warnings

def _score_pore_volume_proxy(linker_feats: dict[str, float | int | list[int]]) -> float:
    rotb = float(linker_feats["NumRotatableBonds"])
    heavy = float(linker_feats["HeavyAtomCount"])
    rotb_score = clamp01(rotb / 12.0)
    heavy_score = clamp01((heavy - 10.0) / 50.0)
    return 0.5 * rotb_score + 0.5 * heavy_score


def _score_symmetry_proxy(raw_smiles: str, cleaned_smiles: str, warnings: list[str]) -> float:
    '''
    Symmetry proxy:
    - preferred: anchor-environment symmetry using [Lr] neighborhoods
    - fallback: topological rank symmetry on cleaned linker'''

    sym_anchor = anchor_env_symmetry_proxy(raw_smiles, env_hops=4, agg="softmin")
    if not math.isnan(sym_anchor):
        return clamp01(sym_anchor)

    sym_topology = topological_symmetry_proxy(cleaned_smiles)
    if not math.isnan(sym_topology):
        warnings.append("Symmetry: used topological fallback (anchor symmetry unavailable)")
        return clamp01(sym_topology)

    warnings.append("Symmetry: unavailable; using neutral default")
    return 0.5


def _score_rigidity(linker_feats: dict[str, float | int | list[int]]) -> float:
    '''Rigid linker proxy: inverse of flexibility (rotatable bond ratio + sp3 fraction)'''
    rot = float(linker_feats["NumRotatableBonds"])
    heavy = max(1.0, float(linker_feats["HeavyAtomCount"]))
    rot_frac = clamp01(rot / heavy)
    frac_csp3 = float(linker_feats["FractionCSP3"])
    flex = 0.5 * rot_frac + 0.5 * clamp01(frac_csp3)
    return clamp01(1.0 - flex)

def _score_alert_penalty(alerts: dict[str, int], penalty_weights: dict[str, float]) -> float:
    '''Normalize alert burden to [0,1] using configured PAINS/BRENK/NIH weights'''

    raw_penalty = (
        penalty_weights["pains"] * alerts["pains"]
        + penalty_weights["brenk"] * alerts["brenk"]
        + penalty_weights["nih"] * alerts["nih"]
    )
    return clamp01(raw_penalty)


def score_candidate(
    linker_smiles: str,
    metal_symbol: str | None = None,
    porosity: dict[str, float] | None = None,
    context: RewardContext | None = None,
    config: RewardConfig | None = None,
) -> RewardResult:
    '''
    Score one MOF linker candidate for drug-delivery suitability.
    ---
    Component design (all scaled to [0,1]):
    1) safety
    2) performance
    3) benign_linkers
    4) stability'''

    config = config or _default_config()
    context = context or build_reward_context()
    warnings: list[str] = []

    if linker_smiles is None or not str(linker_smiles).strip():
        raise ValueError("linker_smiles is required and must be non-empty")

    porosity = porosity or {}
    raw_smiles = str(linker_smiles)
    cleaned_smiles = strip_lr_smiles(raw_smiles)

    # ----------------------------
    # Core descriptors
    # ----------------------------
    feats, feat_warnings = compute_rdkit_features(
        cleaned_smiles,
        include_fingerprints = RDKIT_DEFAULT_INCLUDE_FINGERPRINTS,
        include_3d = RDKIT_DEFAULT_INCLUDE_3D,
    )
    warnings.extend(feat_warnings)
    formal_charge = compute_formal_charge(cleaned_smiles)

    # ----------------------------
    # Safety prerequisites
    # ----------------------------
    metal_score, metal_warn = metal_safety_score(metal_symbol, context.metal_toxicity)
    if metal_warn: warnings.append(metal_warn)

    formal_charge_safety = _score_formal_charge_safety(formal_charge)
    charge_release_risk = 1.0 - formal_charge_safety

    # Toxicity-space safety signal
    tox_prox_non = None
    tox_prox_toxic = None
    cat_prox: dict[int, float] = {}
    tox_safety = None
    s_ghs_proxy = None
    tox_ld50_pred_log10 = float("nan")
    tox_ld50_pred_mgkg = float("nan")
    tox_safety_raw_ld50 = float("nan")
    tox_applicability = float("nan")
    tox_safety_cal = float("nan")
    fragment_risk = float("nan")
    fragment_penalty = float("nan")
    fragment_hit_count = 0
    fragment_hits_json = "[]"
    if context.tox_space is not None:
        feature_vector = np.array([float(feats[f]) for f in context.tox_space.feature_names], dtype=float)
        if np.isnan(feature_vector).any():
            nan_mask = np.isnan(feature_vector)
            feature_vector[nan_mask] = context.tox_space.feature_means[nan_mask]
            warnings.append("Tox proximity: NaN descriptors imputed with tox space means")

        fp_bits_raw = feats.get("ECFP4", [])
        if isinstance(fp_bits_raw, list) and fp_bits_raw:
            fp_bits = [int(x) for x in fp_bits_raw]
        else:
            fp_bits = compute_ecfp4_bits(cleaned_smiles)

        tox_prox_non = tox_proximity_score(context.tox_space, feature_vector, fingerprint_bits=fp_bits)
        cat_prox, tox_prox_toxic, tox_warnings = tox_proximity_by_ghs_category(
            context.tox_space,
            feature_vector,
            fp_bits,
            k=TOX_KNN_K,
            severity_alpha=TOX_SEVERITY_EXP_ALPHA,
        )
        warnings.extend(tox_warnings)
        if 5 in cat_prox and not np.isnan(cat_prox[5]):
            tox_prox_non = cat_prox[5]
        s_ghs_proxy = (
            float(1.0 - tox_prox_toxic)
            if tox_prox_toxic is not None and not np.isnan(tox_prox_toxic)
            else float("nan")
        )

        tox_profile, tox_profile_warnings = tox_ld50_safety_profile(
            context.tox_space,
            feature_vector,
            fp_bits,
            k=TOX_KNN_K,
        )
        warnings.extend(tox_profile_warnings)
        tox_ld50_pred_log10 = float(tox_profile["tox_ld50_pred_log10"])
        tox_ld50_pred_mgkg = float(tox_profile["tox_ld50_pred_mgkg"])
        tox_safety_raw_ld50 = float(tox_profile["tox_safety_raw_ld50"])
        tox_applicability = float(tox_profile["tox_applicability"])
        tox_safety_cal = float(tox_profile["tox_safety_cal"])
        fragment_risk = float(tox_profile["fragment_risk"])
        fragment_penalty = float(tox_profile["fragment_penalty"])
        fragment_hit_count = int(tox_profile["fragment_hit_count"])
        fragment_hits_json = json.dumps(
            tox_profile["fragment_hits"], ensure_ascii=True, sort_keys=True
        )
        tox_safety = float(tox_profile["toxicity_term"])
    else:
        warnings.append("Toxicity feature space missing; tox proximity set to neutral")

    # ----------------------------
    # Performance prerequisites
    # ----------------------------
    length_score = _score_length(raw_smiles, cleaned_smiles, context.linker_stats)
    logp = float(feats["LogP"])
    logs = float(feats["LogS"])
    tpsa = float(feats["TPSA"])
    solubility_score = _score_solubility(logs, logp, config)

    agsa = porosity.get("agsa")
    surface_score, surface_warns = _score_surface_area(agsa, feats, context.mof_stats)
    warnings.extend(surface_warns)

    pld = porosity.get("pld")
    pld_score, pld_warns = _score_pld(pld, length_score, context.mof_stats)
    warnings.extend(pld_warns)
    pore_volume_score = _score_pore_volume_proxy(feats)
    surface_loading_score = 0.5 * surface_score + 0.3 * pore_volume_score + 0.2 * pld_score

    # ----------------------------
    # Benign-linker prerequisites
    # ----------------------------
    alerts = rdkit_alert_counts_from_smiles(cleaned_smiles)
    alert_counts = {"pains": alerts.pains, "brenk": alerts.brenk, "nih": alerts.nih}
    alert_penalty = _score_alert_penalty(alert_counts, config.penalty_weights)
    alert_score = 1.0 - alert_penalty
    payload_interaction_score = _score_payload_interaction(logp, tpsa, config)

    # ----------------------------
    # Stability prerequisites
    # ----------------------------
    symmetry_score = _score_symmetry_proxy(raw_smiles, cleaned_smiles, warnings)
    rigidity_score = _score_rigidity(feats)
    linker_stability = (
        config.linker_stability_subweights["symmetry"] * symmetry_score
        + config.linker_stability_subweights["rigidity"] * rigidity_score
    )
    charge_release_stability = 1.0 - charge_release_risk

    # ----------------------------
    # Component scores (all [0,1])
    # ----------------------------
    safety_tox_term = tox_safety if tox_safety is not None else 0.5
    safety_score = (
        config.safety_subweights["metal"] * metal_score
        + config.safety_subweights["toxicity"] * safety_tox_term
        + config.safety_subweights["formal_charge"] * formal_charge_safety
    )
    safety_score = clamp01(safety_score)

    performance_score = (
        config.performance_subweights["length"] * length_score
        + config.performance_subweights["solubility"] * solubility_score
        + config.performance_subweights["surface_loading"] * surface_loading_score
    )
    performance_score = clamp01(performance_score)

    # NOTE: charge contributes to both safety and benign_linkers by design:
    # safety captures acute electrostatic risk; benign_linkers captures payload interaction fit.
    benign_score = (
        config.benign_subweights["charge"] * formal_charge_safety
        + config.benign_subweights["payload_interaction"] * payload_interaction_score
        + config.benign_subweights["alerts"] * alert_score
    )
    benign_score = clamp01(benign_score)

    stability_score = (
        config.stability_subweights["metal_stability"] * metal_score
        + config.stability_subweights["linker_stability"] * linker_stability
        + config.stability_subweights["solubility"] * solubility_score
        + config.stability_subweights["charge_release"] * charge_release_stability
    )
    stability_score = clamp01(stability_score)

    # Explicit penalty: only extreme charge to avoid double-counting alert penalties.
    penalty = config.penalty_weights["extreme_charge"] * (1.0 if abs(formal_charge) >= 3 else 0.0)
    total = (
        config.component_weights["safety"] * safety_score
        + config.component_weights["performance"] * performance_score
        + config.component_weights["benign_linkers"] * benign_score
        + config.component_weights["stability"] * stability_score
        - penalty
    )
    total = clamp01(total)

    components = {
        "safety": safety_score,
        "performance": performance_score,
        "benign_linkers": benign_score,
        "stability": stability_score,
        "penalty": penalty,
    }

    rdkit_feature_payload: dict[str, object] = {f"rdkit_{k}": v for k, v in feats.items()}

    features: dict[str, object] = {
        "metal_safety": metal_score,
        "linker_length": length_score,
        "symmetry_proxy": symmetry_score,
        "rigidity_proxy": rigidity_score,
        "solubility": solubility_score,
        "benign_linkers": benign_score,
        "payload_interaction_score": payload_interaction_score,
        "formal_charge_safety": formal_charge_safety,
        "logP": logp,
        "logS": logs,
        "TPSA": tpsa,
        "formal_charge": formal_charge,
        "alert_pains": alert_counts["pains"],
        "alert_brenk": alert_counts["brenk"],
        "alert_nih": alert_counts["nih"],
        "alert_penalty": alert_penalty,
        "alert_score": alert_score,
        "tox_proximity_non_toxic": tox_prox_non if tox_prox_non is not None else float("nan"),
        "tox_proximity_toxic": tox_prox_toxic if tox_prox_toxic is not None else float("nan"),
        "tox_proximity_cat1": cat_prox.get(1, float("nan")),
        "tox_proximity_cat2": cat_prox.get(2, float("nan")),
        "tox_proximity_cat3": cat_prox.get(3, float("nan")),
        "tox_proximity_cat4": cat_prox.get(4, float("nan")),
        "tox_proximity_cat5": cat_prox.get(5, float("nan")),
        # Deprecated legacy tox-space proximity fields retained in v1 for compatibility.
        "tox_safety_proxy": safety_tox_term,
        "tox_ld50_pred_log10": tox_ld50_pred_log10,
        "tox_ld50_pred_mgkg": tox_ld50_pred_mgkg,
        "tox_safety_raw_ld50": tox_safety_raw_ld50,
        "tox_applicability": tox_applicability,
        "tox_safety_cal": tox_safety_cal,
        "fragment_risk": fragment_risk,
        "fragment_penalty": fragment_penalty,
        "fragment_hit_count": fragment_hit_count,
        "fragment_hits": fragment_hits_json,
        "surface_area_score": surface_score,
        "pld_score": pld_score,
        "pore_volume_score": pore_volume_score,
        "surface_loading_score": surface_loading_score,
        "performance_score": performance_score,
        "s_charge": formal_charge_safety,
        "s_ghs_proxy": s_ghs_proxy if s_ghs_proxy is not None else float("nan"),
        "metal_stab_proxy": metal_score,
        "linker_stab_proxy": linker_stability,
        "charge_release_risk": charge_release_risk,
        "charge_release_stability": charge_release_stability,
        "safety_score": safety_score,
        "stability_score": stability_score,
        "component_weight_safety": config.component_weights["safety"],
        "component_weight_performance": config.component_weights["performance"],
        "component_weight_benign_linkers": config.component_weights["benign_linkers"],
        "component_weight_stability": config.component_weights["stability"],
        "penalty": penalty,
        "penalty_extreme_charge": penalty,
    }
    features.update(rdkit_feature_payload)

    return RewardResult(
        total_score = total,
        components = components,
        features = features,
        warnings = warnings,
    )


def score_smiles_batch(
    smiles_list: list[str],
    tox_space: DrugToxSpace | None = None,
    *,
    context: RewardContext | None = None,
    config: RewardConfig | None = None,
    skip_invalid: bool | None = None,
    show_progress: bool | None = None,
) -> pd.DataFrame:
    '''
    Score a list of SMILES strings and return a dataframe with:
    - branch_smiles (input)
    - linker_smiles (cleaned)
    - total_score
    - component_* columns
    - feature_* columns
    - warnings and error (when skipped)
    - optional tqdm progress bar (controlled by constants or show_progress arg)'''

    if not isinstance(smiles_list, list):
        raise ValueError("smiles_list must be a list of SMILES strings")
    if skip_invalid is None:
        skip_invalid = REWARD_SKIP_INVALID_SMILES
    if show_progress is None:
        show_progress = REWARD_SHOW_PROGRESS

    context = context or build_reward_context(tox_space=tox_space)
    config = config or _default_config()

    records: list[dict[str, object]] = []
    iterator = enumerate(smiles_list)
    if show_progress and _tqdm is not None:
        iterator = _tqdm(iterator, total=len(smiles_list), desc="Scoring linkers", unit="smiles")
    elif show_progress and _tqdm is None:
        print("Progress requested but tqdm is unavailable; proceeding without progress bar.")

    for idx, raw_smiles in iterator:
        branch_smiles = "" if raw_smiles is None else str(raw_smiles)
        linker_smiles = strip_lr_smiles(branch_smiles) if branch_smiles else ""
        try:
            if not branch_smiles:
                raise ValueError("Empty SMILES")
            result = score_candidate(
                linker_smiles=branch_smiles,
                metal_symbol=None,
                porosity=None,
                context=context,
                config=config,
            )
            rec: dict[str, object] = {
                "row": int(idx),
                "branch_smiles": branch_smiles,
                "linker_smiles": linker_smiles,
                "total_score": float(result.total_score),
                "warnings": "|".join(result.warnings),
            }
            rec.update({f"component_{k}": v for k, v in result.components.items()})
            rec.update({f"feature_{k}": v for k, v in result.features.items()})
        except Exception as e:
            if not skip_invalid:
                raise ValueError(f"Failed to score SMILES at row {idx}: {branch_smiles}") from e
            rec = {
                "row": int(idx),
                "branch_smiles": branch_smiles,
                "linker_smiles": linker_smiles,
                "total_score": math.nan,
                "error": f"{type(e).__name__}: {e}",
            }
        records.append(rec)

    return pd.DataFrame(records)
