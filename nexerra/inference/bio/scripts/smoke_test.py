from __future__ import annotations

import random
import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import LOGS_DIR, MOF_PROPERTIES_COLS
from src.data_loading import load_acute_tox_iv, load_mof_properties
from src.mof_processing import add_clean_linker_smiles
from src.MOF_bio_reward import build_reward_context, score_candidate
from src.utils import init_run_log

# ----------------------------
# Key input variables
# ----------------------------
TOX_CONTEXT_SAMPLE_N: int = 500
MOF_CONTEXT_SAMPLE_N: int = 200


def main() -> None:
    random.seed(42)
    log_path = init_run_log(LOGS_DIR, "smoke_test")
    mof_df_full = load_mof_properties()
    mof_df_full = add_clean_linker_smiles(mof_df_full)
    mof_context_df = mof_df_full.sample(
        n=min(MOF_CONTEXT_SAMPLE_N, len(mof_df_full)),
        random_state=42,
    )

    acute_tox_full = load_acute_tox_iv()
    acute_tox_context = None
    for seed in [42, 43, 44, 45]:
        sample = acute_tox_full.sample(n=min(TOX_CONTEXT_SAMPLE_N, len(acute_tox_full)), random_state=seed)
        try:
            # Ensure both toxic and non-toxic labels exist in sample
            from src.toxicity_classification import classify_ghs

            labels = classify_ghs(sample, ld50_col="Toxicity Value")["ghs_toxic"].unique()
            if len(labels) >= 2:
                acute_tox_context = sample
                break
        except Exception:
            continue
    if acute_tox_context is None:
        acute_tox_context = acute_tox_full

    # Build context (metal toxicity + tox space + normalization stats)
    context = build_reward_context(acute_tox_df=acute_tox_context, mof_properties_df=mof_context_df)

    sample = mof_df_full.sample(n=3, random_state=42)
    lines: list[str] = []
    for _, row in sample.iterrows():
        linker_smiles = row["linker_smiles"]
        porosity = {
            "agsa": row.get(MOF_PROPERTIES_COLS["agsa"]),
            "pld": row.get(MOF_PROPERTIES_COLS["pld"]),
            "lcd": row.get(MOF_PROPERTIES_COLS["lcd"]),
        }
        result = score_candidate(
            linker_smiles=linker_smiles,
            metal_symbol=None,
            porosity=porosity,
            context=context,
        )
        lines.append("---")
        lines.append(f"Linker: {linker_smiles}")
        lines.append(f"Total: {result.total_score:.3f}")
        lines.append(f"Components: {result.components}")
        lines.append(f"Warnings: {result.warnings}")

    with log_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
