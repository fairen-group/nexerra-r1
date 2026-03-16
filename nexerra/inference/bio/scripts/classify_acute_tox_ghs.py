from __future__ import annotations

from pathlib import Path

from src.constants import ACUTE_TOX_COLS, ACUTE_TOX_IV_PATH, INTERIM_DIR, LOGS_DIR
from src.data_loading import load_acute_tox_iv
from src.toxicity_classification import classify_ghs
from src.utils import build_provenance_record, get_git_commit, init_run_log, write_provenance_json

# ----------------------------
# Key input variables
# ----------------------------
OUTPUT_PATH: Path = INTERIM_DIR / "acute_tox_iv_ghs.parquet"
METADATA_PATH: Path = INTERIM_DIR / "acute_tox_iv_ghs.provenance.json"


def main() -> None:
    log_path = init_run_log(LOGS_DIR, "classify_acute_tox_ghs")
    df = load_acute_tox_iv()
    out = classify_ghs(df, ld50_col=ACUTE_TOX_COLS["tox_value"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)

    commit = get_git_commit(Path(__file__).resolve().parents[1])
    record = build_provenance_record([ACUTE_TOX_IV_PATH], code_version=commit)
    write_provenance_json(METADATA_PATH, record)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Saved: {OUTPUT_PATH}\n")
        f.write(f"Rows: {out.shape[0]}\n")
        f.write(f"Metadata: {METADATA_PATH}\n")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
