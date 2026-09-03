from __future__ import annotations

import math
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, GraphDescriptors, Lipinski, MACCSkeys, QED, rdMolDescriptors
from rdkit.Chem import SpacialScore
from rdkit.Chem.rdfiltercatalog import FilterCatalog, FilterCatalogParams

FeatureValue = float | int | list[int] | list[float]
RDKIT_EMBED_SEED = 61453

RDKIT_VECTOR_FEATURES: list[str] = ["GasteigerCharges", "ECFP4", "MACCS"]
RDKIT_SCALAR_FEATURES: list[str] = [
    "MolecularWeight",
    "HeavyAtomMolWt",
    "LogP",
    "MolMR",
    "LogS",
    "TPSA",
    "LabuteASA",
    "BalabanIndex",
    "Chi0",
    "Chi1",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "Phi",
    "FractionCSP3",
    "WienerIndex",
    "BertzCT",
    "MaxEStateIndex",
    "MinEStateIndex",
    "MaxAbsEStateIndex",
    "MinAbsEStateIndex",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumHeterocycles",
    "NumSpiroAtoms",
    "NumBridgeheadAtoms",
    "NumAtomStereoCenters",
    "NumUnspecifiedAtomStereoCenters",
    "HeavyAtomCount",
    "NumHeavyAtoms",
    "NumValenceElectrons",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumHeteroAtoms",
    "NumRadicals",
    "NumRadicalElectrons",
    "NHOHCount",
    "NOCount",
    "TotalOCount",
    "TotalNCount",
    "NumAmideBonds",
    "FpDensityMorgan1",
    "VanDerWaalsVolume",
    "DipoleMoment",
    "QED",
    "LipinskiRO5",
    "FormalCharge",
]
RDKIT_ALL_FEATURES: list[str] = RDKIT_SCALAR_FEATURES + RDKIT_VECTOR_FEATURES
EXPERIMENTALLY_RELEVANT_RDKIT_FEATURES: list[str] = [
    "MolecularWeight",
    "HeavyAtomMolWt",
    "LogP",
    "TPSA",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumHeavyAtoms",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "TotalOCount",
    "TotalNCount",
    "FormalCharge",
]


@dataclass(frozen=True)
class AlertCounts:
    pains: int
    brenk: int
    nih: int


def _make_catalog(which: FilterCatalogParams.FilterCatalogs) -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(which)
    return FilterCatalog(params)


def _get_alert_catalogs() -> tuple[FilterCatalog, FilterCatalog, FilterCatalog]:
    params = FilterCatalogParams.FilterCatalogs
    return (
        _make_catalog(params.PAINS),
        _make_catalog(params.BRENK),
        _make_catalog(params.NIH),
    )


def _safe_number(
    fn: object,
    feature_name: str,
    warnings: list[str],
    cast: type[float] | type[int] = float,
) -> float | int:
    try:
        value = cast(fn())
        if cast is float and not math.isfinite(float(value)):
            warnings.append(f"{feature_name}: non-finite descriptor value ({value}); coerced to NaN")
            return float("nan")
        return value
    except Exception as exc:
        warnings.append(f"{feature_name}: {type(exc).__name__} {exc}")
        return float("nan") if cast is float else 0


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def rdkit_alert_counts(smiles: str) -> AlertCounts:
    mol = mol_from_smiles(smiles)
    pains_cat, brenk_cat, nih_cat = _get_alert_catalogs()
    return AlertCounts(
        pains=len(pains_cat.GetMatches(mol)),
        brenk=len(brenk_cat.GetMatches(mol)),
        nih=len(nih_cat.GetMatches(mol)),
    )


def esol_log_s(mol: Chem.Mol) -> float:
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    return (
        0.16
        - 1.5 * logp
        - 0.01 * (mw - 40.0)
        + 0.066 * rot
        + 0.066 * frac_csp3
        + 0.066 * (tpsa / 100.0)
    )


def _compute_wiener_index(mol: Chem.Mol) -> float:
    dist = Chem.GetDistanceMatrix(mol)
    if dist.size == 0:
        return 0.0
    total = 0.0
    n_atoms = dist.shape[0]
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            total += float(dist[i, j])
    return total


def _compute_gasteiger_charges(mol: Chem.Mol) -> tuple[list[float], str | None]:
    try:
        mol_copy = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(mol_copy)
        charges: list[float] = []
        for atom in mol_copy.GetAtoms():
            raw = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "nan"
            value = float(raw)
            charges.append(value if math.isfinite(value) else float("nan"))
        return charges, None
    except Exception as exc:
        return [], f"GasteigerCharges: {type(exc).__name__} {exc}"


def calc_vdw_volume(mol: Chem.Mol) -> tuple[float, str | None]:
    periodic_table = Chem.GetPeriodicTable()
    missing: list[int] = []
    total = 0.0
    for atom in mol.GetAtoms():
        radius = float(periodic_table.GetRvdw(atom.GetAtomicNum()))
        if radius <= 0:
            missing.append(atom.GetAtomicNum())
            continue
        total += (4.0 / 3.0) * math.pi * (radius ** 3)
    warning = None
    if missing:
        warning = f"VanDerWaalsVolume: missing VdW radii for atomic numbers {sorted(set(missing))}"
    return total, warning


def _compute_dipole_moment(mol: Chem.Mol) -> tuple[float | None, str | None]:
    try:
        mol_h = Chem.AddHs(Chem.Mol(mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = RDKIT_EMBED_SEED
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            return None, "DipoleMoment: 3D conformer embedding failed"
        AllChem.ComputeGasteigerCharges(mol_h)
        conf = mol_h.GetConformer()
        mu_x = 0.0
        mu_y = 0.0
        mu_z = 0.0
        for atom in mol_h.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            charge = float(atom.GetProp("_GasteigerCharge"))
            mu_x += charge * pos.x
            mu_y += charge * pos.y
            mu_z += charge * pos.z
        return float(math.sqrt(mu_x ** 2 + mu_y ** 2 + mu_z ** 2)), None
    except Exception as exc:
        return None, f"DipoleMoment: {type(exc).__name__} {exc}"


def _lipinski_ro5_pass(mol: Chem.Mol) -> int:
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500.0, logp > 5.0, hbd > 5, hba > 10])
    return 1 if violations == 0 else 0


def _compute_n_sps(mol: Chem.Mol, warnings: list[str]) -> float:
    if SpacialScore is None:
        warnings.append("nSPS: unavailable in this RDKit version")
        return float("nan")
    try:
        if mol.GetNumHeavyAtoms() == 0:
            return float("nan")
        return float(SpacialScore.SPS(mol))
    except Exception as exc:
        warnings.append(f"nSPS: {type(exc).__name__} {exc}")
        return float("nan")


def _compute_maccs_bits(mol: Chem.Mol, warnings: list[str]) -> list[int]:
    try:
        if hasattr(rdMolDescriptors, "GetMACCSKeysFingerprint"):
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            return [int(bit) for bit in fp.ToBitString()]
        fp = MACCSkeys.GenMACCSKeys(mol)
        return [int(bit) for bit in fp.ToBitString()]
    except Exception as exc:
        warnings.append(f"MACCS: {type(exc).__name__} {exc}")
        return []


def compute_rdkit_features(
    smiles: str,
    include_fingerprints: bool = False,
    include_3d: bool = True,
) -> tuple[dict[str, FeatureValue], list[str]]:
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")

    mol = mol_from_smiles(smiles)
    warnings: list[str] = []
    features: dict[str, FeatureValue] = {}

    features["MolecularWeight"] = float(Descriptors.MolWt(mol))
    features["HeavyAtomMolWt"] = float(Descriptors.HeavyAtomMolWt(mol))
    features["LogP"] = float(Crippen.MolLogP(mol))
    features["MolMR"] = float(Crippen.MolMR(mol))
    features["LogS"] = float(esol_log_s(mol))
    features["TPSA"] = float(rdMolDescriptors.CalcTPSA(mol))
    features["LabuteASA"] = float(rdMolDescriptors.CalcLabuteASA(mol))
    features["BalabanIndex"] = float(GraphDescriptors.BalabanJ(mol))
    features["Chi0"] = float(GraphDescriptors.Chi0(mol))
    features["Chi1"] = float(GraphDescriptors.Chi1(mol))
    features["Kappa1"] = float(GraphDescriptors.Kappa1(mol))
    features["Kappa2"] = float(GraphDescriptors.Kappa2(mol))
    features["Kappa3"] = float(GraphDescriptors.Kappa3(mol))
    features["Phi"] = _safe_number(lambda: rdMolDescriptors.CalcPhi(mol), "Phi", warnings, cast=float)
    features["FractionCSP3"] = float(rdMolDescriptors.CalcFractionCSP3(mol))
    features["WienerIndex"] = float(_compute_wiener_index(mol))
    features["BertzCT"] = float(Descriptors.BertzCT(mol))
    features["nSPS"] = _compute_n_sps(mol, warnings)
    features["MaxEStateIndex"] = _safe_number(lambda: Descriptors.MaxEStateIndex(mol), "MaxEStateIndex", warnings, cast=float)
    features["MinEStateIndex"] = _safe_number(lambda: Descriptors.MinEStateIndex(mol), "MinEStateIndex", warnings, cast=float)
    features["MaxAbsEStateIndex"] = _safe_number(lambda: Descriptors.MaxAbsEStateIndex(mol), "MaxAbsEStateIndex", warnings, cast=float)
    features["MinAbsEStateIndex"] = _safe_number(lambda: Descriptors.MinAbsEStateIndex(mol), "MinAbsEStateIndex", warnings, cast=float)

    charges, charges_warning = _compute_gasteiger_charges(mol)
    features["GasteigerCharges"] = charges
    if charges_warning:
        warnings.append(charges_warning)
    features["MaxPartialCharge"] = _safe_number(lambda: Descriptors.MaxPartialCharge(mol), "MaxPartialCharge", warnings, cast=float)
    features["MinPartialCharge"] = _safe_number(lambda: Descriptors.MinPartialCharge(mol), "MinPartialCharge", warnings, cast=float)
    features["MaxAbsPartialCharge"] = _safe_number(lambda: Descriptors.MaxAbsPartialCharge(mol), "MaxAbsPartialCharge", warnings, cast=float)
    features["MinAbsPartialCharge"] = _safe_number(lambda: Descriptors.MinAbsPartialCharge(mol), "MinAbsPartialCharge", warnings, cast=float)

    num_radical = _safe_number(lambda: Descriptors.NumRadicalElectrons(mol), "NumRadicalElectrons", warnings, cast=int)
    features["RingCount"] = int(rdMolDescriptors.CalcNumRings(mol))
    features["NumAromaticRings"] = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    features["NumAliphaticRings"] = int(rdMolDescriptors.CalcNumAliphaticRings(mol))
    features["NumSaturatedRings"] = int(rdMolDescriptors.CalcNumSaturatedRings(mol))
    features["NumAromaticCarbocycles"] = int(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
    features["NumAromaticHeterocycles"] = int(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
    features["NumAliphaticCarbocycles"] = int(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
    features["NumAliphaticHeterocycles"] = int(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
    features["NumSaturatedCarbocycles"] = int(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))
    features["NumSaturatedHeterocycles"] = int(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))
    features["NumHeterocycles"] = int(rdMolDescriptors.CalcNumHeterocycles(mol))
    features["NumSpiroAtoms"] = int(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    features["NumBridgeheadAtoms"] = int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    features["NumAtomStereoCenters"] = _safe_number(lambda: rdMolDescriptors.CalcNumAtomStereoCenters(mol), "NumAtomStereoCenters", warnings, cast=int)
    features["NumUnspecifiedAtomStereoCenters"] = _safe_number(
        lambda: rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol),
        "NumUnspecifiedAtomStereoCenters",
        warnings,
        cast=int,
    )
    features["HeavyAtomCount"] = _safe_number(lambda: Descriptors.HeavyAtomCount(mol), "HeavyAtomCount", warnings, cast=int)
    features["NumHeavyAtoms"] = int(mol.GetNumHeavyAtoms())
    features["NumValenceElectrons"] = _safe_number(lambda: Descriptors.NumValenceElectrons(mol), "NumValenceElectrons", warnings, cast=float)
    features["NumHDonors"] = int(Lipinski.NumHDonors(mol))
    features["NumHAcceptors"] = int(Lipinski.NumHAcceptors(mol))
    features["NumRotatableBonds"] = int(Lipinski.NumRotatableBonds(mol))
    features["NumHeteroAtoms"] = _safe_number(lambda: rdMolDescriptors.CalcNumHeteroatoms(mol), "NumHeteroAtoms", warnings, cast=int)
    features["NumRadicals"] = num_radical
    features["NumRadicalElectrons"] = num_radical
    features["NHOHCount"] = int(Lipinski.NHOHCount(mol))
    features["NOCount"] = int(Lipinski.NOCount(mol))
    features["TotalOCount"] = int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8))
    features["TotalNCount"] = int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7))
    features["NumAmideBonds"] = _safe_number(lambda: rdMolDescriptors.CalcNumAmideBonds(mol), "NumAmideBonds", warnings, cast=int)
    features["FpDensityMorgan1"] = _safe_number(lambda: Descriptors.FpDensityMorgan1(mol), "FpDensityMorgan1", warnings, cast=float)
    vdw_volume, vdw_warning = calc_vdw_volume(mol)
    features["VanDerWaalsVolume"] = float(vdw_volume)
    if vdw_warning:
        warnings.append(vdw_warning)
    if include_3d:
        dipole, dipole_warning = _compute_dipole_moment(mol)
        features["DipoleMoment"] = float("nan") if dipole is None else float(dipole)
        if dipole_warning:
            warnings.append(dipole_warning)
    else:
        features["DipoleMoment"] = float("nan")
    features["QED"] = float(QED.qed(mol))
    features["LipinskiRO5"] = _lipinski_ro5_pass(mol)
    features["FormalCharge"] = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))

    if include_fingerprints:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        features["ECFP4"] = [int(bit) for bit in fp.ToBitString()]
        features["MACCS"] = _compute_maccs_bits(mol, warnings)
    else:
        features["ECFP4"] = []
        features["MACCS"] = []

    return features, warnings
