from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from src.constants import (
    ACUTE_TOX_COLS,
    ACUTE_TOX_IP_PATH,
    ACUTE_TOX_IV_PATH,
    DRUG_TOX_FRAGMENTS_PATH,
    DRUG_TOX_SPACE_FULL_PATH,
    DRUG_TOX_SPACE_JSON_PATH,
    LOGS_DIR,
)
from src.data_loading import load_acute_tox_ip, load_acute_tox_iv
from src.drug_tox_space import (
    build_drug_tox_space,
    build_pooled_acute_tox_df,
    save_drug_tox_space,
)
from src.utils import build_provenance_record, get_git_commit, init_run_log, write_provenance_json

# ----------------------------
# Key input variables
# ----------------------------
SPACE_JSON_PATH: Path = DRUG_TOX_SPACE_JSON_PATH
SPACE_FULL_PATH: Path = DRUG_TOX_SPACE_FULL_PATH
FRAGMENTS_CSV_PATH: Path = DRUG_TOX_FRAGMENTS_PATH
METADATA_PATH: Path = SPACE_JSON_PATH.with_suffix(".provenance.json")
SHOW_PROGRESS: bool = True


def main() -> None:
    log_path = init_run_log(LOGS_DIR, "build_drug_tox_space")
    iv_df = load_acute_tox_iv()
    ip_df = load_acute_tox_ip()
    pooled_df = build_pooled_acute_tox_df(
        iv_df,
        ip_df,
        smiles_col=ACUTE_TOX_COLS["canonical_smiles"],
        ld50_col=ACUTE_TOX_COLS["tox_value"],
        inchikey_col=ACUTE_TOX_COLS["inchikey"],
    )

    space = build_drug_tox_space(
        pooled_df,
        smiles_col=ACUTE_TOX_COLS["canonical_smiles"],
        ld50_col=ACUTE_TOX_COLS["tox_value"],
        inchikey_col=ACUTE_TOX_COLS["inchikey"],
        dedupe_by_identity=False,
        show_progress=SHOW_PROGRESS,
    )

    SPACE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_drug_tox_space(space, SPACE_FULL_PATH)
    with SPACE_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": space.feature_names,
                "feature_means": space.feature_means.tolist(),
                "feature_stds": space.feature_stds.tolist(),
                "non_toxic_centroid": space.non_toxic_centroid.tolist(),
                "toxic_centroid": space.toxic_centroid.tolist(),
                "distance_scale": space.distance_scale,
                "fp_non_toxic_centroid": space.fp_non_toxic_centroid.tolist(),
                "fp_toxic_centroid": space.fp_toxic_centroid.tolist(),
                "block_weights": space.block_weights,
                "train_size": int(len(space.train_log_ld50)),
            },
            f,
            indent=2,
        )

    frag_rows = [
        {
            "bit_id": bit_id,
            "effect": float(space.bit_effect.get(bit_id, float("nan"))),
            "support": int(space.bit_support.get(bit_id, 0)),
            "log_odds": float(space.bit_log_odds.get(bit_id, float("nan"))),
            "fragment_smiles": frag,
        }
        for bit_id, frag in space.bit_fragments.items()
    ]
    frag_df = pd.DataFrame(frag_rows)
    if not frag_df.empty:
        frag_df = frag_df.sort_values(
            by=["effect", "support"], ascending=[True, False], kind="mergesort"
        )
    frag_df.to_csv(FRAGMENTS_CSV_PATH, index=False)

    commit = get_git_commit(Path(__file__).resolve().parents[1])
    record = build_provenance_record(
        [ACUTE_TOX_IV_PATH, ACUTE_TOX_IP_PATH], code_version=commit
    )
    write_provenance_json(METADATA_PATH, record)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Saved: {SPACE_FULL_PATH}\n")
        f.write(f"Saved: {SPACE_JSON_PATH}\n")
        f.write(f"Saved: {FRAGMENTS_CSV_PATH}\n")
        f.write(
            "Pooled acute tox rows: "
            f"iv={len(iv_df)}, ip={len(ip_df)}, pooled_deduped={len(pooled_df)}\n"
        )
        f.write(f"LD50 train rows: {len(space.train_log_ld50)}\n")
        f.write(f"Fragments: {len(space.bit_fragments)}\n")
        f.write(f"Metadata: {METADATA_PATH}\n")
    print(f"Saved: {SPACE_FULL_PATH}")
    print(f"Saved: {SPACE_JSON_PATH}")
    print(f"Saved: {FRAGMENTS_CSV_PATH}")


if __name__ == "__main__":
    main()
