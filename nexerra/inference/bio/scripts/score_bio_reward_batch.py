from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import (
    DRUG_TOX_SPACE_FULL_PATH,
    INTERIM_DIR,
    REWARD_CONTEXT_STATS_PATH,
    REWARD_SKIP_INVALID_SMILES,
)
from src.MOF_bio_reward import build_reward_context, load_reusable_reward_context, score_smiles_batch

# ----------------------------
# Key input variables
# ----------------------------
DEFAULT_TOX_SPACE_PATH: Path = DRUG_TOX_SPACE_FULL_PATH
DEFAULT_OUTPUT_PATH: Path = INTERIM_DIR / "generated_linker_scores.parquet"
DEFAULT_SMILES_COL: str = "smiles"
CONTEXT_CHOICES: tuple[str, str] = ("reuse", "build")


def _dedupe_smiles(smiles_list: list[str]) -> tuple[list[str], dict[str, int]]:
    """
    Deduplicate SMILES while preserving first-seen order.

    Returns:
    - unique_smiles: ordered unique SMILES
    - counts: occurrence count per SMILES in original input
    """
    counts: dict[str, int] = {}
    unique_smiles: list[str] = []
    for smi in smiles_list:
        key = str(smi).strip()
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == 1:
            unique_smiles.append(key)
    return unique_smiles, counts


def _load_smiles_from_file(path: Path, smiles_col: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".smi", ".smiles"}:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [line for line in lines if line]

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported input format. Use .txt/.smi/.csv/.parquet.")

    if smiles_col not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_col}' not found in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    values = df[smiles_col].astype(str).tolist()
    return [v for v in values if v and v.strip()]


def _build_context(context_mode: str, tox_space_path: Path) -> object:
    if context_mode == "build":
        context = build_reward_context()
        print(f"Context stats source: {context.stats_source}")
        print(f"Context linker_stats: {context.linker_stats}")
        print(f"Context mof_stats: {context.mof_stats}")
        print(f"Saved context stats artifact: {REWARD_CONTEXT_STATS_PATH}")
        return context
    try:
        context = load_reusable_reward_context(tox_space_path=tox_space_path)
        print(f"Context stats source: {context.stats_source}")
        print(f"Context linker_stats: {context.linker_stats}")
        print(f"Context mof_stats: {context.mof_stats}")
        print(f"Saved context stats artifact: {REWARD_CONTEXT_STATS_PATH}")
        return context
    except ValueError as e:
        raise ValueError(
            f"Failed to load tox context from {tox_space_path}. "
            "Build it first with `python scripts/build_drug_tox_space.py` "
            "or use --context-mode build."
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a list of linker SMILES and write parquet output."
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--smiles",
        action="append",
        default=None,
        help="SMILES value. Repeat this flag for multiple entries.",
    )
    src_group.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Input file containing SMILES (.txt/.smi/.csv/.parquet).",
    )
    parser.add_argument(
        "--smiles-col",
        default=DEFAULT_SMILES_COL,
        help=f"Column name for SMILES in csv/parquet input (default: {DEFAULT_SMILES_COL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--context-mode",
        choices=CONTEXT_CHOICES,
        default="reuse",
        help="`reuse` loads precomputed tox space, `build` recomputes context from raw data.",
    )
    parser.add_argument(
        "--tox-space-path",
        type=Path,
        default=DEFAULT_TOX_SPACE_PATH,
        help=f"Path to precomputed tox space NPZ (default: {DEFAULT_TOX_SPACE_PATH}).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Fail on first invalid SMILES instead of skipping invalid rows.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar output.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable deduplication; score all input rows (including duplicates).",
    )
    args = parser.parse_args()

    if args.input_file is not None:
        smiles_list = _load_smiles_from_file(args.input_file, args.smiles_col)
    else:
        smiles_list = [s for s in (args.smiles or []) if s and s.strip()]

    if not smiles_list:
        raise ValueError("No SMILES provided for batch scoring.")

    if args.no_dedupe:
        smiles_to_score = smiles_list
        duplicate_counts = {s: 1 for s in smiles_list}
        print(f"Scoring all input rows (dedupe disabled): {len(smiles_to_score)}")
    else:
        smiles_to_score, duplicate_counts = _dedupe_smiles(smiles_list)
        print(
            "Deduplicated SMILES before scoring: "
            f"{len(smiles_list)} input -> {len(smiles_to_score)} unique"
        )

    context = _build_context(args.context_mode, args.tox_space_path)
    skip_invalid = REWARD_SKIP_INVALID_SMILES if not args.fail_fast else False
    out_df = score_smiles_batch(
        smiles_list=smiles_to_score,
        context=context,
        skip_invalid=skip_invalid,
        show_progress=not args.no_progress,
    )
    out_df["duplicate_count"] = (
        out_df["branch_smiles"].map(duplicate_counts).fillna(1).astype(int)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"Saved batch scores: {args.output} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
