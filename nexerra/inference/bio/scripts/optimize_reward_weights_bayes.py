from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from rdkit import RDLogger

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import (
    ACUTE_TOX_COLS,
    DRUG_TOX_SPACE_FULL_PATH,
    INTERIM_DIR,
    LOGS_DIR,
    PENALTY_WEIGHTS,
    REWARD_COMPONENT_WEIGHTS,
    SAFETY_SUBWEIGHTS,
    PERFORMANCE_SUBWEIGHTS,
    BENIGN_LINKER_SUBWEIGHTS,
    STABILITY_SUBWEIGHTS,
    LINKER_STABILITY_SUBWEIGHTS,
    TOX_SEVERITY_EXP_ALPHA,
)
from src.MOF_bio_reward import (
    RewardContext,
    build_reward_context,
    load_reusable_reward_context,
    score_candidate,
)
from src.data_loading import load_acute_tox_iv
from src.toxicity_classification import classify_ghs
from src.utils import clamp01, init_run_log

# ----------------------------
# Key input variables
# ----------------------------
DEFAULT_OUTPUT_JSON: Path = INTERIM_DIR / "optimized_reward_weights_bayes.json"
DEFAULT_OUTPUT_PARQUET: Path = INTERIM_DIR / "optimized_reward_weights_eval.parquet"
DEFAULT_TOX_SPACE_PATH: Path = DRUG_TOX_SPACE_FULL_PATH
DEFAULT_CONTEXT_MODE: str = "reuse"
CONTEXT_CHOICES: tuple[str, str] = ("reuse", "build")
DEFAULT_CATEGORIES: tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_SAMPLE_PER_CATEGORY: int = 25
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_INIT_POINTS: int = 20
DEFAULT_BAYES_ITERS: int = 30
DEFAULT_ACQ_CANDIDATES: int = 3000
DEFAULT_MARGIN: float = 0.02
DEFAULT_BOUNDS: tuple[float, float] = (-2.5, 2.5)
GP_LENGTH_SCALE: float = 1.20
GP_NOISE: float = 1e-6
DEFAULT_ORDER_PENALTY: float = 2.0
DEFAULT_CAT5_BONUS: float = 0.2

# Parameterized blocks: optimize (n - 1) logits and anchor the last at 0.
COMPONENT_KEYS: tuple[str, ...] = ("safety", "performance", "benign_linkers", "stability")
SAFETY_KEYS: tuple[str, ...] = ("metal", "toxicity", "formal_charge")
PERFORMANCE_KEYS: tuple[str, ...] = ("length", "solubility", "surface_loading")
BENIGN_KEYS: tuple[str, ...] = ("charge", "payload_interaction", "alerts")
STABILITY_KEYS: tuple[str, ...] = ("metal_stability", "linker_stability", "solubility", "charge_release")
LINKER_STABILITY_KEYS: tuple[str, ...] = ("symmetry", "rigidity")
SEARCH_DIMENSIONS: int = (
    len(COMPONENT_KEYS)
    - 1
    + len(SAFETY_KEYS)
    - 1
    + len(PERFORMANCE_KEYS)
    - 1
    + len(BENIGN_KEYS)
    - 1
    + len(STABILITY_KEYS)
    - 1
    + len(LINKER_STABILITY_KEYS)
    - 1
)


@dataclass(frozen=True)
class SampleAtoms:
    smiles: str
    ghs_category: int
    metal_score: float
    tox_safety: float
    formal_charge_safety: float
    length_score: float
    solubility_score: float
    surface_loading_score: float
    payload_interaction_score: float
    alert_score: float
    symmetry_score: float
    rigidity_score: float
    charge_release_stability: float
    extreme_charge_flag: int


@dataclass(frozen=True)
class WeightBundle:
    component_weights: dict[str, float]
    safety_subweights: dict[str, float]
    performance_subweights: dict[str, float]
    benign_subweights: dict[str, float]
    stability_subweights: dict[str, float]
    linker_stability_subweights: dict[str, float]


@dataclass(frozen=True)
class ObjectiveMetrics:
    objective: float
    tox_weighted_mean: float
    ranking_penalty: float
    category_means: dict[int, float]
    cat1_lt_cat4: bool
    cat1_lt_cat5: bool


def _log(msg: str, log_path: Path) -> None:
    print(msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


def _parse_categories(raw: str) -> list[int]:
    try:
        cats = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except Exception as e:  # pragma: no cover - parse guard
        raise ValueError(f"Invalid --categories value: {raw}") from e
    if not cats:
        raise ValueError("At least one category is required.")
    unknown = [c for c in cats if c < 1 or c > 5]
    if unknown:
        raise ValueError(f"Categories must be between 1 and 5. Got: {unknown}")
    return cats


def _build_context(context_mode: str, tox_space_path: Path) -> RewardContext:
    if context_mode == "build":
        return build_reward_context()
    return load_reusable_reward_context(tox_space_path=tox_space_path)


def _softmax_block(logits: np.ndarray, keys: tuple[str, ...]) -> dict[str, float]:
    if logits.ndim != 1:
        raise ValueError("Logits must be a 1D vector.")
    if logits.shape[0] != len(keys) - 1:
        raise ValueError(
            f"Expected {len(keys) - 1} logits for keys={keys}, got {logits.shape[0]}"
        )
    anchored = np.concatenate([logits.astype(float), np.array([0.0], dtype=float)])
    anchored -= np.max(anchored)
    probs = np.exp(anchored)
    probs /= probs.sum()
    return {k: float(probs[i]) for i, k in enumerate(keys)}


def _vector_to_weights(vector: np.ndarray) -> WeightBundle:
    if vector.ndim != 1 or vector.shape[0] != SEARCH_DIMENSIONS:
        raise ValueError(
            f"Weight vector shape mismatch: expected ({SEARCH_DIMENSIONS},), got {vector.shape}"
        )

    cursor = 0

    def take(n: int) -> np.ndarray:
        nonlocal cursor
        chunk = vector[cursor : cursor + n]
        cursor += n
        return chunk

    component_weights = _softmax_block(take(len(COMPONENT_KEYS) - 1), COMPONENT_KEYS)
    safety_subweights = _softmax_block(take(len(SAFETY_KEYS) - 1), SAFETY_KEYS)
    performance_subweights = _softmax_block(
        take(len(PERFORMANCE_KEYS) - 1), PERFORMANCE_KEYS
    )
    benign_subweights = _softmax_block(take(len(BENIGN_KEYS) - 1), BENIGN_KEYS)
    stability_subweights = _softmax_block(take(len(STABILITY_KEYS) - 1), STABILITY_KEYS)
    linker_stability_subweights = _softmax_block(
        take(len(LINKER_STABILITY_KEYS) - 1), LINKER_STABILITY_KEYS
    )

    if cursor != SEARCH_DIMENSIONS:
        raise ValueError(
            f"Internal cursor mismatch: consumed {cursor} of {SEARCH_DIMENSIONS} dimensions."
        )

    return WeightBundle(
        component_weights=component_weights,
        safety_subweights=safety_subweights,
        performance_subweights=performance_subweights,
        benign_subweights=benign_subweights,
        stability_subweights=stability_subweights,
        linker_stability_subweights=linker_stability_subweights,
    )


def _default_weight_bundle() -> WeightBundle:
    return WeightBundle(
        component_weights=dict(REWARD_COMPONENT_WEIGHTS),
        safety_subweights=dict(SAFETY_SUBWEIGHTS),
        performance_subweights=dict(PERFORMANCE_SUBWEIGHTS),
        benign_subweights=dict(BENIGN_LINKER_SUBWEIGHTS),
        stability_subweights=dict(STABILITY_SUBWEIGHTS),
        linker_stability_subweights=dict(LINKER_STABILITY_SUBWEIGHTS),
    )


def _score_from_atoms(sample: SampleAtoms, weights: WeightBundle) -> float:
    linker_stability = (
        weights.linker_stability_subweights["symmetry"] * sample.symmetry_score
        + weights.linker_stability_subweights["rigidity"] * sample.rigidity_score
    )
    safety_score = (
        weights.safety_subweights["metal"] * sample.metal_score
        + weights.safety_subweights["toxicity"] * sample.tox_safety
        + weights.safety_subweights["formal_charge"] * sample.formal_charge_safety
    )
    performance_score = (
        weights.performance_subweights["length"] * sample.length_score
        + weights.performance_subweights["solubility"] * sample.solubility_score
        + weights.performance_subweights["surface_loading"] * sample.surface_loading_score
    )
    benign_score = (
        weights.benign_subweights["charge"] * sample.formal_charge_safety
        + weights.benign_subweights["payload_interaction"]
        * sample.payload_interaction_score
        + weights.benign_subweights["alerts"] * sample.alert_score
    )
    stability_score = (
        weights.stability_subweights["metal_stability"] * sample.metal_score
        + weights.stability_subweights["linker_stability"] * linker_stability
        + weights.stability_subweights["solubility"] * sample.solubility_score
        + weights.stability_subweights["charge_release"] * sample.charge_release_stability
    )
    penalty = PENALTY_WEIGHTS["extreme_charge"] * float(sample.extreme_charge_flag)
    total = (
        weights.component_weights["safety"] * safety_score
        + weights.component_weights["performance"] * performance_score
        + weights.component_weights["benign_linkers"] * benign_score
        + weights.component_weights["stability"] * stability_score
        - penalty
    )
    return clamp01(total)


def _evaluate_weights(
    samples: list[SampleAtoms],
    weights: WeightBundle,
    margin: float,
    order_penalty: float,
    cat5_bonus_weight: float,
) -> ObjectiveMetrics:
    by_cat: dict[int, list[float]] = defaultdict(list)
    for sample in samples:
        by_cat[sample.ghs_category].append(_score_from_atoms(sample, weights))

    cat_means: dict[int, float] = {}
    for cat, vals in by_cat.items():
        if vals:
            cat_means[cat] = float(np.mean(vals))

    weighted_sum = 0.0
    weight_total = 0.0
    for cat in [1, 2, 3, 4]:
        if cat not in cat_means:
            continue
        cat_weight = float(np.exp(TOX_SEVERITY_EXP_ALPHA * (5 - cat)))
        weighted_sum += cat_weight * cat_means[cat]
        weight_total += cat_weight
    if weight_total <= 0.0:
        raise ValueError("No category 1-4 samples were available for objective evaluation.")
    tox_weighted_mean = weighted_sum / weight_total

    cat1 = cat_means.get(1, math.nan)
    cat4 = cat_means.get(4, math.nan)
    cat5 = cat_means.get(5, math.nan)
    cat1_lt_cat4 = bool(np.isfinite(cat1) and np.isfinite(cat4) and cat1 < cat4)
    cat1_lt_cat5 = bool(np.isfinite(cat1) and np.isfinite(cat5) and cat1 < cat5)

    ranking_penalty = 0.0
    if np.isfinite(cat1) and np.isfinite(cat4):
        ranking_penalty += max(0.0, margin - (cat4 - cat1))
    if np.isfinite(cat1) and np.isfinite(cat5):
        ranking_penalty += max(0.0, margin - (cat5 - cat1))
    cat5_bonus = cat5 if np.isfinite(cat5) else 0.0

    objective = (
        tox_weighted_mean
        + order_penalty * ranking_penalty
        - cat5_bonus_weight * cat5_bonus
    )
    return ObjectiveMetrics(
        objective=float(objective),
        tox_weighted_mean=float(tox_weighted_mean),
        ranking_penalty=float(ranking_penalty),
        category_means=cat_means,
        cat1_lt_cat4=cat1_lt_cat4,
        cat1_lt_cat5=cat1_lt_cat5,
    )


def _is_finite_sample(value: float) -> bool:
    return bool(np.isfinite(value))


def _extract_atoms(smiles: str, ghs_category: int, context: RewardContext) -> SampleAtoms | None:
    result = score_candidate(linker_smiles=smiles, context=context)
    feat = result.features

    atoms = SampleAtoms(
        smiles=smiles,
        ghs_category=int(ghs_category),
        metal_score=float(feat["metal_safety"]),
        tox_safety=float(feat["tox_safety_proxy"]),
        formal_charge_safety=float(feat["formal_charge_safety"]),
        length_score=float(feat["linker_length"]),
        solubility_score=float(feat["solubility"]),
        surface_loading_score=float(feat["surface_loading_score"]),
        payload_interaction_score=float(feat["payload_interaction_score"]),
        alert_score=float(feat["alert_score"]),
        symmetry_score=float(feat["symmetry_proxy"]),
        rigidity_score=float(feat["rigidity_proxy"]),
        charge_release_stability=float(feat["charge_release_stability"]),
        extreme_charge_flag=int(abs(int(feat["formal_charge"])) >= 3),
    )
    finite_checks = [
        atoms.metal_score,
        atoms.tox_safety,
        atoms.formal_charge_safety,
        atoms.length_score,
        atoms.solubility_score,
        atoms.surface_loading_score,
        atoms.payload_interaction_score,
        atoms.alert_score,
        atoms.symmetry_score,
        atoms.rigidity_score,
        atoms.charge_release_stability,
    ]
    if not all(_is_finite_sample(v) for v in finite_checks):
        return None
    return atoms


def _collect_sample_atoms(
    labeled_df: pd.DataFrame,
    smiles_col: str,
    categories: list[int],
    per_category: int,
    seed: int,
    context: RewardContext,
    log_path: Path,
) -> list[SampleAtoms]:
    rng = np.random.default_rng(seed)
    atoms: list[SampleAtoms] = []

    for cat in categories:
        cat_df = labeled_df[labeled_df["ghs_category"] == cat]
        smiles_values = (
            cat_df[smiles_col].dropna().astype(str).str.strip().replace("", np.nan).dropna()
        )
        unique_smiles = smiles_values.drop_duplicates().tolist()
        if not unique_smiles:
            _log(f"[WARN] Category {cat}: no usable SMILES.", log_path)
            continue

        order = rng.permutation(len(unique_smiles))
        kept = 0
        attempted = 0
        for idx in order:
            attempted += 1
            smi = unique_smiles[int(idx)]
            try:
                row = _extract_atoms(smi, cat, context)
            except Exception:
                row = None
            if row is None:
                continue
            atoms.append(row)
            kept += 1
            if kept >= per_category:
                break
        _log(
            f"Category {cat}: kept {kept} samples (target={per_category}, attempted={attempted})",
            log_path,
        )

    if not atoms:
        raise ValueError("No valid samples available for optimization.")
    return atoms


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "erf"):
        return 0.5 * (1.0 + np.erf(x / math.sqrt(2.0)))
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    sqdist = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * sqdist / (length_scale * length_scale))


@dataclass(frozen=True)
class GPModel:
    x_train: np.ndarray
    chol: np.ndarray
    alpha: np.ndarray
    y_mean: float
    y_std: float
    length_scale: float


def _fit_gp(x_train: np.ndarray, y_train: np.ndarray) -> GPModel:
    if x_train.ndim != 2:
        raise ValueError("x_train must be 2D.")
    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D.")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train/y_train row mismatch.")

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-12:
        y_std = 1.0
    y_scaled = (y_train - y_mean) / y_std

    k = _rbf_kernel(x_train, x_train, GP_LENGTH_SCALE)
    eye = np.eye(k.shape[0], dtype=float)
    chol = None
    for jitter in (0.0, 1e-8, 1e-6, 1e-4):
        try:
            chol = np.linalg.cholesky(k + (GP_NOISE + jitter) * eye)
            break
        except np.linalg.LinAlgError:
            continue
    if chol is None:
        raise ValueError("Failed to fit GP: kernel matrix is not positive definite.")

    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_scaled))
    return GPModel(
        x_train=x_train,
        chol=chol,
        alpha=alpha,
        y_mean=y_mean,
        y_std=y_std,
        length_scale=GP_LENGTH_SCALE,
    )


def _predict_gp(model: GPModel, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x_test.ndim != 2:
        raise ValueError("x_test must be 2D.")
    k_trans = _rbf_kernel(model.x_train, x_test, model.length_scale)
    mu_scaled = k_trans.T @ model.alpha
    v = np.linalg.solve(model.chol, k_trans)
    var_scaled = np.maximum(1.0 - np.sum(v * v, axis=0), 1e-12)
    mu = mu_scaled * model.y_std + model.y_mean
    sigma = np.sqrt(var_scaled) * model.y_std
    return mu, sigma


def _expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best_observed: float,
    xi: float = 0.01,
) -> np.ndarray:
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = (best_observed - mu - xi)
    z = improvement / sigma_safe
    ei = improvement * _normal_cdf(z) + sigma_safe * _normal_pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def _run_bayesian_optimization(
    objective_fn: Callable[[np.ndarray], float],
    seed: int,
    n_init: int,
    n_iter: int,
    n_candidates: int,
    bounds: tuple[float, float],
    log_path: Path,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    if n_init < 2:
        raise ValueError("n_init must be >= 2")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")

    low, high = bounds
    rng = np.random.default_rng(seed)

    x_hist: list[np.ndarray] = []
    y_hist: list[float] = []
    history: list[dict[str, float]] = []

    for i in range(n_init):
        x = rng.uniform(low, high, size=SEARCH_DIMENSIONS)
        y = objective_fn(x)
        x_hist.append(x)
        y_hist.append(y)
        best_so_far = float(np.min(np.array(y_hist, dtype=float)))
        history.append({"phase": "init", "step": float(i), "objective": y, "best": best_so_far})
        _log(f"[init {i + 1}/{n_init}] objective={y:.6f} best={best_so_far:.6f}", log_path)

    for i in range(n_iter):
        x_arr = np.vstack(x_hist).astype(float)
        y_arr = np.array(y_hist, dtype=float)
        model = _fit_gp(x_arr, y_arr)

        global_candidates = rng.uniform(low, high, size=(n_candidates, SEARCH_DIMENSIONS))
        best_idx = int(np.argmin(y_arr))
        local_candidates = x_arr[best_idx] + rng.normal(
            loc=0.0,
            scale=0.35,
            size=(max(200, n_candidates // 5), SEARCH_DIMENSIONS),
        )
        local_candidates = np.clip(local_candidates, low, high)
        candidates = np.vstack([global_candidates, local_candidates])

        mu, sigma = _predict_gp(model, candidates)
        acquisition = _expected_improvement(mu, sigma, best_observed=float(np.min(y_arr)))
        next_idx = int(np.argmax(acquisition))
        x_next = candidates[next_idx]

        if np.any(np.all(np.isclose(x_arr, x_next, atol=1e-9), axis=1)):
            x_next = rng.uniform(low, high, size=SEARCH_DIMENSIONS)

        y_next = objective_fn(x_next)
        x_hist.append(x_next)
        y_hist.append(y_next)

        best_so_far = float(np.min(np.array(y_hist, dtype=float)))
        history.append(
            {
                "phase": "bayes",
                "step": float(i),
                "objective": float(y_next),
                "best": best_so_far,
            }
        )
        _log(
            f"[bayes {i + 1}/{n_iter}] objective={y_next:.6f} best={best_so_far:.6f}",
            log_path,
        )

    y_final = np.array(y_hist, dtype=float)
    x_final = np.vstack(x_hist).astype(float)
    best_idx = int(np.argmin(y_final))
    return x_final[best_idx], float(y_final[best_idx]), history


def _weights_to_dict(weights: WeightBundle) -> dict[str, dict[str, float]]:
    return {
        "REWARD_COMPONENT_WEIGHTS": weights.component_weights,
        "SAFETY_SUBWEIGHTS": weights.safety_subweights,
        "PERFORMANCE_SUBWEIGHTS": weights.performance_subweights,
        "BENIGN_LINKER_SUBWEIGHTS": weights.benign_subweights,
        "STABILITY_SUBWEIGHTS": weights.stability_subweights,
        "LINKER_STABILITY_SUBWEIGHTS": weights.linker_stability_subweights,
    }


def _metrics_to_dict(metrics: ObjectiveMetrics) -> dict[str, object]:
    cat_means = {f"cat_{k}": float(v) for k, v in sorted(metrics.category_means.items())}
    return {
        "objective": float(metrics.objective),
        "tox_weighted_mean": float(metrics.tox_weighted_mean),
        "ranking_penalty": float(metrics.ranking_penalty),
        "cat1_lt_cat4": bool(metrics.cat1_lt_cat4),
        "cat1_lt_cat5": bool(metrics.cat1_lt_cat5),
        "category_means": cat_means,
    }


def _build_eval_dataframe(
    samples: list[SampleAtoms],
    default_weights: WeightBundle,
    optimized_weights: WeightBundle,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample in samples:
        rows.append(
            {
                "smiles": sample.smiles,
                "ghs_category": int(sample.ghs_category),
                "score_default": float(_score_from_atoms(sample, default_weights)),
                "score_optimized": float(_score_from_atoms(sample, optimized_weights)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bayesian optimization for REWARD_COMPONENT_WEIGHTS and SUBWEIGHTS "
            "using IV acute-toxicity SMILES. Objective penalizes high scores in "
            "cats 1-4 with exponential severity scaling."
        )
    )
    parser.add_argument(
        "--context-mode",
        choices=CONTEXT_CHOICES,
        default=DEFAULT_CONTEXT_MODE,
        help="`reuse` loads precomputed tox space. `build` recomputes from raw data.",
    )
    parser.add_argument(
        "--tox-space-path",
        type=Path,
        default=DEFAULT_TOX_SPACE_PATH,
        help=f"Path to precomputed tox-space NPZ (default: {DEFAULT_TOX_SPACE_PATH}).",
    )
    parser.add_argument(
        "--categories",
        default="1,2,3,4,5",
        help="Comma-separated GHS categories to sample (default: 1,2,3,4,5).",
    )
    parser.add_argument(
        "--sample-per-category",
        type=int,
        default=DEFAULT_SAMPLE_PER_CATEGORY,
        help=f"Target valid SMILES per category (default: {DEFAULT_SAMPLE_PER_CATEGORY}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for deterministic sampling and optimization (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--init-points",
        type=int,
        default=DEFAULT_INIT_POINTS,
        help=f"Initial random evaluations before BO (default: {DEFAULT_INIT_POINTS}).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_BAYES_ITERS,
        help=f"Bayesian optimization iterations (default: {DEFAULT_BAYES_ITERS}).",
    )
    parser.add_argument(
        "--acq-candidates",
        type=int,
        default=DEFAULT_ACQ_CANDIDATES,
        help=f"Acquisition candidate points per BO step (default: {DEFAULT_ACQ_CANDIDATES}).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"Required margin for cat1 < cat4 and cat1 < cat5 checks (default: {DEFAULT_MARGIN}).",
    )
    parser.add_argument(
        "--order-penalty",
        type=float,
        default=DEFAULT_ORDER_PENALTY,
        help=(
            "Objective weight for ranking-margin violations "
            f"(default: {DEFAULT_ORDER_PENALTY})."
        ),
    )
    parser.add_argument(
        "--cat5-bonus",
        type=float,
        default=DEFAULT_CAT5_BONUS,
        help=(
            "Objective bonus weight for cat5 mean score "
            f"(default: {DEFAULT_CAT5_BONUS})."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON for optimized weights/metrics (default: {DEFAULT_OUTPUT_JSON}).",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
        help=f"Output parquet with per-sample default/optimized scores (default: {DEFAULT_OUTPUT_PARQUET}).",
    )
    parser.add_argument(
        "--require-ordering",
        action="store_true",
        help="Exit with code 1 if optimized weights do not satisfy cat1 < cat4 and cat1 < cat5.",
    )
    args = parser.parse_args()

    log_path = init_run_log(LOGS_DIR, "optimize_reward_weights_bayes")
    _log(f"Run log: {log_path}", log_path)

    # Silence noisy parser warnings to keep optimization logs readable.
    RDLogger.DisableLog("rdApp.*")
    warnings.filterwarnings(
        "ignore",
        message="Skipped .* linker rows .* failed RDKit parsing.*",
        category=RuntimeWarning,
    )

    categories = _parse_categories(args.categories)
    _log(f"Using categories: {categories}", log_path)
    _log(f"Search dimensions: {SEARCH_DIMENSIONS}", log_path)

    iv_df = load_acute_tox_iv()
    labeled = classify_ghs(iv_df, ld50_col=ACUTE_TOX_COLS["tox_value"])
    smiles_col = ACUTE_TOX_COLS["canonical_smiles"]
    _log(f"Loaded IV dataset: rows={len(labeled)}", log_path)

    context = _build_context(args.context_mode, args.tox_space_path)
    _log(f"Context mode: {args.context_mode}", log_path)

    samples = _collect_sample_atoms(
        labeled_df=labeled,
        smiles_col=smiles_col,
        categories=categories,
        per_category=args.sample_per_category,
        seed=args.seed,
        context=context,
        log_path=log_path,
    )
    cat_counts: dict[int, int] = defaultdict(int)
    for sample in samples:
        cat_counts[sample.ghs_category] += 1
    _log(f"Collected valid samples: {len(samples)}", log_path)
    _log(f"Category counts: {dict(sorted(cat_counts.items()))}", log_path)

    default_weights = _default_weight_bundle()
    baseline = _evaluate_weights(
        samples,
        default_weights,
        margin=args.margin,
        order_penalty=args.order_penalty,
        cat5_bonus_weight=args.cat5_bonus,
    )
    _log(f"Baseline objective: {baseline.objective:.6f}", log_path)
    _log(f"Baseline category means: {baseline.category_means}", log_path)

    eval_cache: dict[tuple[float, ...], ObjectiveMetrics] = {}

    def objective(vector: np.ndarray) -> float:
        key = tuple(np.round(vector.astype(float), 8).tolist())
        if key in eval_cache:
            return eval_cache[key].objective
        weights = _vector_to_weights(vector)
        metrics = _evaluate_weights(
            samples,
            weights,
            margin=args.margin,
            order_penalty=args.order_penalty,
            cat5_bonus_weight=args.cat5_bonus,
        )
        eval_cache[key] = metrics
        return metrics.objective

    best_vector, best_objective, history = _run_bayesian_optimization(
        objective_fn=objective,
        seed=args.seed,
        n_init=args.init_points,
        n_iter=args.iters,
        n_candidates=args.acq_candidates,
        bounds=DEFAULT_BOUNDS,
        log_path=log_path,
    )
    best_weights = _vector_to_weights(best_vector)
    best_metrics = _evaluate_weights(
        samples,
        best_weights,
        margin=args.margin,
        order_penalty=args.order_penalty,
        cat5_bonus_weight=args.cat5_bonus,
    )

    _log(f"Best objective: {best_objective:.6f}", log_path)
    _log(f"Optimized category means: {best_metrics.category_means}", log_path)
    _log(
        f"Ordering checks: cat1<cat4={best_metrics.cat1_lt_cat4}, cat1<cat5={best_metrics.cat1_lt_cat5}",
        log_path,
    )

    out_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(Path("data/raw") / "Acute Toxicity_mouse_intravenous_LD50_with_units.csv"),
        "settings": {
            "categories": categories,
            "sample_per_category": int(args.sample_per_category),
            "seed": int(args.seed),
            "init_points": int(args.init_points),
            "iters": int(args.iters),
            "acq_candidates": int(args.acq_candidates),
            "margin": float(args.margin),
            "order_penalty": float(args.order_penalty),
            "cat5_bonus": float(args.cat5_bonus),
            "tox_severity_exp_alpha": float(TOX_SEVERITY_EXP_ALPHA),
        },
        "sample_counts": {f"cat_{k}": int(v) for k, v in sorted(cat_counts.items())},
        "baseline_metrics": _metrics_to_dict(baseline),
        "optimized_metrics": _metrics_to_dict(best_metrics),
        "optimized_weights": _weights_to_dict(best_weights),
        "baseline_weights": _weights_to_dict(default_weights),
        "search_history": history,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    _log(f"Wrote JSON: {args.output_json}", log_path)

    eval_df = _build_eval_dataframe(samples, default_weights, best_weights)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_parquet(args.output_parquet, index=False)
    _log(f"Wrote parquet: {args.output_parquet} (rows={len(eval_df)})", log_path)

    if args.require_ordering and not (
        best_metrics.cat1_lt_cat4 and best_metrics.cat1_lt_cat5
    ):
        raise SystemExit(
            "Ordering check failed: expected cat1 mean score < cat4 and cat5 means."
        )


if __name__ == "__main__":
    main()
