from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import (
    ACUTE_TOX_COLS,
    ACUTE_TOX_IV_PATH,
    INTERIM_DIR,
    LOGS_DIR,
    RDKIT_DEFAULT_INCLUDE_3D,
    RDKIT_DEFAULT_INCLUDE_FINGERPRINTS,
)
from src.data_loading import load_acute_tox_iv
from src.feature_extraction import featurize_smiles_df
from src.toxicity_classification import classify_ghs
from src.utils import build_provenance_record, get_git_commit, init_run_log, write_provenance_json

# ----------------------------
# Key input variables
# ----------------------------
OUTPUT_PATH: Path = INTERIM_DIR / "acute_tox_iv_rdkit_features.parquet"
METADATA_PATH: Path = INTERIM_DIR / "acute_tox_iv_rdkit_features.provenance.json"
SHOW_PROGRESS: bool = True


def main() -> None:
    log_path = init_run_log(LOGS_DIR, "extract_rdkit_features")
    acute_tox_df = load_acute_tox_iv()
    acute_tox_ghs_df = classify_ghs(acute_tox_df, ld50_col=ACUTE_TOX_COLS["tox_value"])
    rdkit_df, warnings = featurize_smiles_df(
        acute_tox_ghs_df,
        smiles_col=ACUTE_TOX_COLS["canonical_smiles"],
        include_fingerprints=RDKIT_DEFAULT_INCLUDE_FINGERPRINTS,
        include_3d=RDKIT_DEFAULT_INCLUDE_3D,
        include_smiles=False,
        show_progress=SHOW_PROGRESS,
    )
    out = pd.concat([acute_tox_ghs_df.reset_index(drop=True), rdkit_df.reset_index(drop=True)], axis=1)

    if out.shape[0] != acute_tox_ghs_df.shape[0]:
        raise ValueError("Row count changed after appending RDKit features")

    if not set(acute_tox_ghs_df.columns).issubset(set(out.columns)):
        raise ValueError("acute_tox_ghs_df columns are missing in output parquet")

    out.to_parquet(OUTPUT_PATH, index=False)
    commit = get_git_commit(Path(__file__).resolve().parents[1])
    record = build_provenance_record([ACUTE_TOX_IV_PATH], code_version=commit)
    write_provenance_json(METADATA_PATH, record)
    warn_count = sum(bool(w) for w in warnings)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Saved: {OUTPUT_PATH}\n")
        f.write(f"Rows: {out.shape[0]}\n")
        f.write(f"Warnings: {warn_count}\n")
        f.write(f"Metadata: {METADATA_PATH}\n")
    print(f"Saved: {OUTPUT_PATH} (warnings for {warn_count} rows)")


if __name__ == "__main__":
    main()
