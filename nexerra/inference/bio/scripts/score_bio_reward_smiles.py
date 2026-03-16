from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import DRUG_TOX_SPACE_FULL_PATH, REWARD_CONTEXT_STATS_PATH
from src.MOF_bio_reward import build_reward_context, load_reusable_reward_context, score_candidate

# ----------------------------
# Key input variables
# ----------------------------
DEFAULT_TOX_SPACE_PATH: Path = DRUG_TOX_SPACE_FULL_PATH
PRINT_CHOICES: tuple[str, str, str] = ("score", "components", "full")
CONTEXT_CHOICES: tuple[str, str] = ("reuse", "build")


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


def _format_output(mode: str, result: object) -> str:
    total_score = float(result.total_score)
    if mode == "score":
        return f"{total_score:.6f}"
    if mode == "components":
        payload = {
            "total_score": total_score,
            "components": result.components,
        }
        return json.dumps(payload, indent=2, allow_nan=True)
    payload = {
        "total_score": total_score,
        "components": result.components,
        "features": result.features,
        "warnings": result.warnings,
    }
    return json.dumps(payload, indent=2, allow_nan=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one linker SMILES for bio reward and print selected output view."
    )
    parser.add_argument("--smiles", required=True, help="Input linker SMILES.")
    parser.add_argument("--print", dest="print_mode", choices=PRINT_CHOICES,
        default="score", help="Output mode: score | components | full.")
    parser.add_argument(
        "--context-mode",
        choices=CONTEXT_CHOICES,
        default="reuse",
        help="`reuse` loads precomputed tox space, `build` recomputes context from raw data.",
    )
    parser.add_argument(
        "--tox-space-path", type=Path, default=DEFAULT_TOX_SPACE_PATH, help=f"Path to precomputed tox space NPZ (default: {DEFAULT_TOX_SPACE_PATH}).")
    parser.add_argument("--metal-symbol", default=None, help="Optional metal symbol (e.g., Zn).")
    parser.add_argument("--agsa", type=float, default=None, help="Optional AGSA porosity value.")
    parser.add_argument("--pld", type=float, default=None, help="Optional PLD porosity value.")
    parser.add_argument("--lcd", type=float, default=None, help="Optional LCD porosity value.")
    args = parser.parse_args()

    context = _build_context(args.context_mode, args.tox_space_path)
    porosity = {"agsa": args.agsa, "pld": args.pld, "lcd": args.lcd}
    result = score_candidate(
        linker_smiles=args.smiles,
        metal_symbol=args.metal_symbol,
        porosity=porosity,
        context=context,
    )
    print(_format_output(args.print_mode, result))


if __name__ == "__main__":
    main()
