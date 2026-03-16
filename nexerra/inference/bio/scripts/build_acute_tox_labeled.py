from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import INTERIM_DIR, LOGS_DIR
from src.toxicity_data import create_acute_tox_labeled_df
from src.utils import init_run_log

# ----------------------------
# Key input variables
# ----------------------------
OUTPUT_PATH: Path = INTERIM_DIR / "acute_tox_iv_labeled.parquet"
METADATA_PATH: Path = INTERIM_DIR / "acute_tox_iv_labeled.provenance.json"


def main() -> None:
    log_path = init_run_log(LOGS_DIR, "build_acute_tox_labeled")
    labeled = create_acute_tox_labeled_df(save_path=OUTPUT_PATH, metadata_path=METADATA_PATH)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Saved: {OUTPUT_PATH}\n")
        f.write(f"Rows: {labeled.shape[0]}\n")
        f.write(f"Metadata: {METADATA_PATH}\n")
    print(f"Saved labeled tox dataset: {OUTPUT_PATH} (rows={labeled.shape[0]})")


if __name__ == "__main__":
    main()
