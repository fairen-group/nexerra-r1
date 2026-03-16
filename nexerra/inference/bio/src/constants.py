from __future__ import annotations

from pathlib import Path

# ----------------------------
# Project paths
# ----------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# ----------------------------
# Raw data files
# ----------------------------
ACUTE_TOX_IV_FILENAME: str = "Acute Toxicity_mouse_intravenous_LD50_with_units.csv"
ACUTE_TOX_IP_FILENAME: str = "Acute Toxicity_mouse_intraperitoneal_LD50_with_units.csv"
# MOF_SMILES_FILENAME: str = "mof_properties_train.csv"
MOF_SMILES_FILENAME: str = "train_smiles.csv"
MOF_METAL_TOXICITY_FILENAME: str = "MOF_metal_toxicity.csv"

ACUTE_TOX_IV_PATH: Path = RAW_DIR / ACUTE_TOX_IV_FILENAME
ACUTE_TOX_IP_PATH: Path = RAW_DIR / ACUTE_TOX_IP_FILENAME
MOF_SMILES_PATH: Path = RAW_DIR / MOF_SMILES_FILENAME
MOF_METAL_TOXICITY_PATH: Path = RAW_DIR / MOF_METAL_TOXICITY_FILENAME

# ----------------------------
# Reusable derived artifacts
# ----------------------------
DRUG_TOX_SPACE_JSON_FILENAME: str = "drug_tox_space.json"
DRUG_TOX_SPACE_FULL_FILENAME: str = "drug_tox_space_full.npz"
DRUG_TOX_FRAGMENTS_FILENAME: str = "drug_tox_fragments.csv"

DRUG_TOX_SPACE_JSON_PATH: Path = PROCESSED_DIR / DRUG_TOX_SPACE_JSON_FILENAME
DRUG_TOX_SPACE_FULL_PATH: Path = PROCESSED_DIR / DRUG_TOX_SPACE_FULL_FILENAME
DRUG_TOX_FRAGMENTS_PATH: Path = PROCESSED_DIR / DRUG_TOX_FRAGMENTS_FILENAME
REWARD_CONTEXT_STATS_FILENAME: str = "reward_context_stats.json"
REWARD_CONTEXT_STATS_PATH: Path = PROCESSED_DIR / REWARD_CONTEXT_STATS_FILENAME

# ----------------------------
# Column names
# ----------------------------
ACUTE_TOX_COLS: dict[str, str] = {
    "taid": "TAID",
    "name": "Name",
    "iupac_name": "IUPAC Name",
    "pubchem_cid": "PubChem CID",
    "canonical_smiles": "Canonical SMILES",
    "inchikey": "InChIKey",
    "tox_value": "Toxicity Value",
}

MOF_PROPERTIES_COLS: dict[str, str] = {
    "organic_core": "organic_core",
    "metal_node": "metal_node",
    "topology": "topology",
    "branch_smiles": "smiles",
    "lcd": "lcd",
    "pld": "pld",
    "density": "density",
    "agsa": "agsa",
}

METAL_TOXICITY_COLS: dict[str, str] = {
    "metal": "Metal",
    "toxicity": "Toxicity",
    "reference": "Reference",
    "confidence": "Confidence",
}

# ----------------------------
# GHS classification thresholds
# ----------------------------
# NOTE: These thresholds are the standard GHS acute toxicity oral cutoffs (mg/kg).
# For IV/IP data, treat as a proxy until route-specific thresholds are defined.
GHS_LD50_THRESHOLDS_MGKG: list[float] = [5.0, 50.0, 300.0, 2000.0, 5000.0]
# Numeric GHS categories (1-5). 0 is reserved for unclassified.
GHS_CATEGORY_LABELS: list[int] = [1, 2, 3, 4, 5]
GHS_TOXIC_CATEGORIES: set[int] = {1, 2, 3, 4}

# ----------------------------
# Toxicity space (k-NN severity)
# ----------------------------
# k for per-category k-NN proximity in tox space
TOX_KNN_K: int = 15
# Exponential severity scaling: higher alpha increases cat1 weight vs cat4
TOX_SEVERITY_EXP_ALPHA: float = 0.6
# Applicability threshold for emitting OOD safety warnings.
TOX_LOW_APPLICABILITY_WARNING_THRESHOLD: float = 0.20
# Neutral toxicity safety baseline used when applicability is low.
TOX_OOD_NEUTRAL_BASELINE: float = 0.50
# Fragment-risk settings for toxicity explainability and mild penalty.
TOX_FRAGMENT_MIN_SUPPORT: int = 100
TOX_FRAGMENT_MAX_PENALTY: float = 0.10
TOX_FRAGMENT_MAX_HITS: int = 12
TOX_FRAGMENT_MAX_MAP: int = 250

# ----------------------------
# RDKit feature list (user-specified)
# ----------------------------
# Centralized RDKit extraction toggles
RDKIT_DEFAULT_INCLUDE_FINGERPRINTS: bool = True
RDKIT_DEFAULT_INCLUDE_3D: bool = True # 3D features can be expensive to compute.

# Each entry includes a short descriptor summary.
RDKIT_FEATURES: list[str] = [
    "MolecularWeight",                 # MolWt (Da): size/volume proxy.
    "HeavyAtomMolWt",                  # Heavy-atom molecular weight.
    "LogP",                            # Crippen cLogP: hydrophobicity.
    "MolMR",                           # Molar refractivity proxy.
    "LogS",                            # ESOL logS proxy: solubility.
    "TPSA",                            # Topological polar surface area.
    "LabuteASA",                       # Approximate solvent-accessible area.
    "BalabanIndex",                    # Balaban J topological index.
    "Chi0",                            # Connectivity index Chi0.
    "Chi1",                            # Connectivity index Chi1.
    "Kappa1",                          # Kier shape index 1.
    "Kappa2",                          # Kier shape index 2.
    "Kappa3",                          # Kier shape index 3.
    "Phi",                             # Kier flexibility/shape descriptor.
    "FractionCSP3",                    # sp3 fraction: 3D character proxy.
    "WienerIndex",                     # Sum of shortest-path distances.
    "BertzCT",                         # Topological complexity index.
    # "nSPS",                            # Normalized spatial score (if available).
    "MaxEStateIndex",                  # Max E-state descriptor.
    "MinEStateIndex",                  # Min E-state descriptor.
    "MaxAbsEStateIndex",               # Max absolute E-state descriptor.
    "MinAbsEStateIndex",               # Min absolute E-state descriptor.
    "GasteigerCharges",                # Per-atom Gasteiger partial charges.
    "MaxPartialCharge",                # Max atom partial charge.
    "MinPartialCharge",                # Min atom partial charge.
    "MaxAbsPartialCharge",             # Max absolute atom partial charge.
    "MinAbsPartialCharge",             # Min absolute atom partial charge.
    "RingCount",                       # Total ring count.
    "NumAromaticRings",                # Aromatic ring count.
    "NumAliphaticRings",               # Aliphatic ring count.
    "NumSaturatedRings",               # Saturated ring count.
    "NumAromaticCarbocycles",          # Aromatic C-only rings.
    "NumAromaticHeterocycles",         # Aromatic heterocycle count.
    "NumAliphaticCarbocycles",         # Aliphatic C-only rings.
    "NumAliphaticHeterocycles",        # Aliphatic heterocycle count.
    "NumSaturatedCarbocycles",         # Saturated C-only rings.
    "NumSaturatedHeterocycles",        # Saturated heterocycle count.
    "NumHeterocycles",                 # Total heterocycle count.
    "NumSpiroAtoms",                   # Spiro atom count.
    "NumBridgeheadAtoms",              # Bridgehead atom count.
    "NumAtomStereoCenters",            # Assigned stereocenter count.
    "NumUnspecifiedAtomStereoCenters", # Unspecified stereocenter count.
    "HeavyAtomCount",                  # Heavy atom count.
    "NumValenceElectrons",             # Valence electron count.
    "NumHDonors",                      # H-bond donor count.
    "NumHAcceptors",                   # H-bond acceptor count.
    "NumRotatableBonds",               # Rotatable bond count.
    "NumHeteroAtoms",                  # Hetero atom count.
    "NumRadicals",                     # Radical electron count.
    "NumRadicalElectrons",             # Explicit radical-electron descriptor.
    "NHOHCount",                       # N-H and O-H motif count.
    "NOCount",                         # N and O atom count.
    "NumAmideBonds",                   # Amide bond count.
    "FpDensityMorgan1",                # Morgan bit density (r=1).
    "VanDerWaalsVolume",               # Summed atom VdW-sphere volume.
    "DipoleMoment",                    # Gasteiger dipole (3D conformer).
    "QED",                             # Drug-likeness score.
    "LipinskiRO5",                     # RO5 pass/fail (0/1).
    "ECFP4",                           # Morgan fingerprint (r=2, 2048 bits).
    "MACCS",                           # MACCS key fingerprint bits.
]

# Numeric subset for toxicity feature space (exclude fingerprints)
TOX_SPACE_NUMERIC_FEATURES: list[str] = [
    "MolecularWeight",                 # Size proxy.
    "HeavyAtomMolWt",                  # Heavy-atom mass proxy.
    "LogP",                            # Hydrophobicity.
    "MolMR",                           # Polarizability/volume proxy.
    "LogS",                            # Solubility proxy.
    "TPSA",                            # Polarity.
    "LabuteASA",                       # Surface-area proxy.
    "BalabanIndex",                    # Topological connectivity.
    "FractionCSP3",                    # 3D character.
    "Chi0",                            # Connectivity index.
    "Chi1",                            # Connectivity index.
    "Kappa1",                          # Shape index.
    "Kappa2",                          # Shape index.
    "Kappa3",                          # Shape index.
    "Phi",                             # Flexibility/shape descriptor.
    "WienerIndex",                     # Size/branching.
    "BertzCT",                         # Complexity.
    # "nSPS",                            # Normalized spatial score (if available).
    "MaxEStateIndex",                  # Electronic state extreme.
    "MinEStateIndex",                  # Electronic state extreme.
    "MaxAbsEStateIndex",               # Electronic state magnitude.
    "MinAbsEStateIndex",               # Electronic state magnitude.
    "MaxPartialCharge",                # Charge extreme.
    "MinPartialCharge",                # Charge extreme.
    "MaxAbsPartialCharge",             # Charge magnitude.
    "MinAbsPartialCharge",             # Charge magnitude.
    "VanDerWaalsVolume",               # VDW sphere sum.
    "RingCount",                       # Ring count.
    "NumAromaticRings",                # Aromatic rings.
    "NumAliphaticRings",               # Aliphatic rings.
    "NumSaturatedRings",               # Saturated rings.
    "NumAromaticCarbocycles",          # Aromatic C-only rings.
    "NumAromaticHeterocycles",         # Aromatic heterocycles.
    "NumAliphaticCarbocycles",         # Aliphatic C-only rings.
    "NumAliphaticHeterocycles",        # Aliphatic heterocycles.
    "NumSaturatedCarbocycles",         # Saturated C-only rings.
    "NumSaturatedHeterocycles",        # Saturated heterocycles.
    "NumHeterocycles",                 # Total heterocycles.
    "NumSpiroAtoms",                   # Spiro centers.
    "NumBridgeheadAtoms",              # Bridgehead atoms.
    "NumAtomStereoCenters",            # Defined stereocenters.
    "NumUnspecifiedAtomStereoCenters", # Undefined stereocenters.
    "HeavyAtomCount",                  # Heavy atom count.
    "NumValenceElectrons",             # Valence electrons.
    "NumHDonors",                      # H-bond donors.
    "NumHAcceptors",                   # H-bond acceptors.
    "NumRotatableBonds",               # Flexibility proxy.
    "NumHeteroAtoms",                  # Hetero atom count.
    "NumRadicals",                     # Radical electrons.
    "NHOHCount",                       # NH/OH motif count.
    "NOCount",                         # N+O atom count.
    "NumAmideBonds",                   # Amide motif count.
    "FpDensityMorgan1",                # Fingerprint bit density.
    "QED",                             # Drug-likeness.
    "LipinskiRO5",                     # RO5 pass/fail.
    "DipoleMoment",                    # 3D dipole (optional).
]

# ----------------------------
# Reward weights (top-level components)
# ----------------------------
# Safety, performance, benign_linkers, stability must sum to 1.0.
REWARD_COMPONENT_WEIGHTS: dict[str, float] = {
    "safety": 0.60,
    "performance": 0.14,
    "benign_linkers": 0.07,
    "stability": 0.19,
}

# Backward-compatible alias used by existing modules.
REWARD_BLOCK_WEIGHTS: dict[str, float] = REWARD_COMPONENT_WEIGHTS

# Component-level internal blend weights (each dict sums to 1.0).
SAFETY_SUBWEIGHTS: dict[str, float] = {
    "metal": 0.10,
    "toxicity": 0.80,
    "formal_charge": 0.10,
}

PERFORMANCE_SUBWEIGHTS: dict[str, float] = {
    "length": 0.15,
    "solubility": 0.10,
    "surface_loading": 0.75,
}

BENIGN_LINKER_SUBWEIGHTS: dict[str, float] = {
    "charge": 0.10,
    "payload_interaction": 0.60,
    "alerts": 0.30,
}

STABILITY_SUBWEIGHTS: dict[str, float] = {
    "metal_stability": 0.25,
    "linker_stability": 0.10,
    "solubility": 0.30,
    "charge_release": 0.35,
}

LINKER_STABILITY_SUBWEIGHTS: dict[str, float] = {
    "symmetry": 0.10,
    "rigidity": 0.90,
}

# Reward descriptor windows and ranges
REWARD_SOLUBILITY_LOGS_RANGE: tuple[float, float] = (-6.0, 0.0)
REWARD_LOGP_WINDOW: tuple[float, float] = (0.5, 3.0)
REWARD_TPSA_WINDOW: tuple[float, float] = (40.0, 120.0)

# Reward execution defaults
REWARD_SKIP_INVALID_SMILES: bool = True
REWARD_SHOW_PROGRESS: bool = True

# Reward context stats mode:
# - True  => use fixed stats below (fast; preferred for iterative single-smiles scoring loops)
# - False => fit stats from provided MOF dataset
REWARD_USE_FIXED_CONTEXT_STATS: bool = True

# Force dataset re-fit of mof_stats + linker_stats even if fixed stats mode is enabled.
# Use this to refresh stats values, then copy updated values back into the fixed placeholders below.
REWARD_FORCE_RECALCULATE_CONTEXT_STATS: bool = False

# Fixed stats placeholders (update after a refresh run if needed).
REWARD_FIXED_MOF_STATS: dict[str, tuple[float, float]] = {
    "agsa": (0.0, 9463.0),
    "pld": (0.0, 101.0),
    "lcd": (0.3, 117.0),
}
REWARD_FIXED_LINKER_STATS: dict[str, tuple[float, float]] = {
    "heavy_atom_count": (1.0, 50.0),
    "anchor_distance": (1.45, 33.35),
}

# Penalty weights
PENALTY_WEIGHTS: dict[str, float] = {
    "pains": 0.10,
    "brenk": 0.20,
    "nih": 0.20,
    "extreme_charge": 0.30,
}

# Deterministic seed used for RDKit conformer generation (when needed)
RDKIT_EMBED_SEED: int = 42


