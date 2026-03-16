# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
#  This module implements a biocompatibility reward function compatible with the style
#  of Reward.py, while using a deterministic, auditable data pipeline.
#  Author: Ivan Zyuzin
#  --- 
#  
#  Please note: Parts of this code are still under-development [...]
#  MIT License
#  Copyright (c) 2026 The Authors
# ---------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Any
import math

from .constants import RDKIT_DEFAULT_INCLUDE_3D, RDKIT_DEFAULT_INCLUDE_FINGERPRINTS, RDKIT_EMBED_SEED, RDKIT_FEATURES


from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import (
        AllChem,
        Crippen,
        Descriptors,
        GraphDescriptors,
        Lipinski,
        QED,
        rdMolDescriptors)
from rdkit.Chem.rdfiltercatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem import MACCSkeys
from rdkit.Chem import SpacialScore

FeatureValue = float | int | list[int] | list[float]

def _ensure_rdkit() -> None:
    return True

@dataclass(frozen = True)
class AlertCounts:
    pains: int
    brenk: int
    nih: int

def _make_catalog(which: Any) -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(which)
    return FilterCatalog(params)

def _get_alert_catalogs() -> tuple[FilterCatalog, FilterCatalog, FilterCatalog]:
    _ensure_rdkit()
    params = FilterCatalogParams.FilterCatalogs
    return (
        _make_catalog(params.PAINS),
        _make_catalog(params.BRENK),
        _make_catalog(params.NIH),
    )

def _safe_number(
    fn: Any,
    feature_name: str,
    warnings: list[str],
    cast: type[float] | type[int] = float,
) -> float | int:
    '''Safely evaluate a descriptor function and return NaN/0 with warning on failure'''
    try:
        value = cast(fn())
        if cast is float and not math.isfinite(float(value)):
            warnings.append(
                f"{feature_name}: non-finite descriptor value ({value}); coerced to NaN"
            )
            return float("nan")
        return value
    except Exception as e: 
        warnings.append(f"{feature_name}: {type(e).__name__} {e}")
        return float("nan") if cast is float else 0


def mol_from_smiles(smiles: str) -> "Chem.Mol":
    '''Parse a SMILES string into an RDKit Mol'''
    _ensure_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
    return mol

def rdkit_alert_counts(mol: "Chem.Mol") -> AlertCounts:
    '''Count PAINS/BRENK/NIH alert matches for a molecule'''
    _ensure_rdkit()
    pains_cat, brenk_cat, nih_cat = _get_alert_catalogs()
    return AlertCounts(
        pains = len(pains_cat.GetMatches(mol)),
        brenk = len(brenk_cat.GetMatches(mol)),
        nih = len(nih_cat.GetMatches(mol)),
    )


def rdkit_alert_counts_from_smiles(smiles: str) -> AlertCounts:
    '''Count PAINS/BRENK/NIH alert matches for a SMILES string'''
    mol = mol_from_smiles(smiles)
    return rdkit_alert_counts(mol)


def compute_ecfp4_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> list[int]:
    '''Compute ECFP4 (Morgan radius 2) fingerprint bits for a SMILES string'''
    _ensure_rdkit()
    mol = mol_from_smiles(smiles)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = radius, nBits = n_bits)
    return [int(x) for x in fp.ToBitString()]


def esol_logS(mol: "Chem.Mol") -> float:
    '''Compute ESOL-like logS proxy (higher is more soluble)'''
    _ensure_rdkit()
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


def _compute_wiener_index(mol: "Chem.Mol") -> float:
    '''Compute Wiener index as the sum of all pairwise shortest path lengths'''
    _ensure_rdkit()
    dist = Chem.GetDistanceMatrix(mol)
    if dist.size == 0: return 0.0
    total = 0.0
    n = dist.shape[0]
    for i in range(n):
        for j in range(i + 1, n): total += float(dist[i, j])
    return total


def _compute_gasteiger_charges(mol: "Chem.Mol") -> tuple[list[float], str | None]:
    '''Compute per-atom Gasteiger partial charges on the 2D graph'''
    _ensure_rdkit()
    try:
        mol_copy = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(mol_copy)
        charges: list[float] = []
        for atom in mol_copy.GetAtoms():
            raw = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "nan"
            value = float(raw)
            if math.isnan(value) or math.isinf(value):
                value = float("nan")
            charges.append(value)
        return charges, None
    except Exception as e: 
        return [], f"GasteigerCharges: {type(e).__name__} {e}"


def calc_vdw_volume(mol: "Chem.Mol") -> tuple[float, str | None]:
    '''Approximate van der Waals volume using summed atom spheres'''

    _ensure_rdkit()
    pt = Chem.GetPeriodicTable()
    missing: list[int] = []
    total = 0.0
    for atom in mol.GetAtoms():
        r = float(pt.GetRvdw(atom.GetAtomicNum()))
        if r <= 0:
            missing.append(atom.GetAtomicNum())
            continue
        total += (4.0 / 3.0) * math.pi * (r**3)

    warning = None
    if missing:
        warning = f"VanDerWaalsVolume: missing VdW radii for atomic numbers {sorted(set(missing))}"
    return total, warning


def _compute_dipole_moment(mol: "Chem.Mol") -> tuple[float | None, str | None]:
    '''Estimate dipole moment from one 3D conformer + Gasteiger charges'''
    _ensure_rdkit()
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

        dipole = math.sqrt(mu_x**2 + mu_y**2 + mu_z**2)
        return float(dipole), None
    except Exception as e: 
        return None, f"DipoleMoment: {type(e).__name__} {e}"

def _lipinski_ro5_pass(mol: "Chem.Mol") -> int:
    '''Return 1 if molecule passes basic Lipinski RO5 cutoffs'''
    _ensure_rdkit()
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500.0, logp > 5.0, hbd > 5, hba > 10])
    return 1 if violations == 0 else 0


def _compute_n_sps(mol: "Chem.Mol", warnings: list[str]) -> float:
    '''Compute normalized spatial score if available in this RDKit build'''
    if SpacialScore is None:
        warnings.append("nSPS: unavailable in this RDKit version")
        return float("nan")
    try:
        if mol.GetNumHeavyAtoms() == 0:
            return float("nan")
        return float(SpacialScore.SPS(mol))
    except Exception as e:  # pragma: no cover - version dependent
        warnings.append(f"nSPS: {type(e).__name__} {e}")
        return float("nan")


def _compute_maccs_bits(mol: "Chem.Mol", warnings: list[str]) -> list[int]:
    '''Compute MACCS keys with fallback across RDKit builds'''
    try:
        if hasattr(rdMolDescriptors, "GetMACCSKeysFingerprint"):
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            return [int(x) for x in fp.ToBitString()]
        if MACCSkeys is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            return [int(x) for x in fp.ToBitString()]
    except Exception as e:
        warnings.append(f"MACCS: {type(e).__name__} {e}")
        return []
    warnings.append("MACCS: unavailable in this RDKit build")
    return []

def compute_rdkit_features(
    smiles: str,
    include_fingerprints: bool | None = None,
    include_3d: bool | None = None,
) -> tuple[dict[str, FeatureValue], list[str]]:
    '''
    Compute RDKit descriptors for a SMILES string
    ---
    Returns:
    - features: descriptor dictionary
    - warnings: descriptor-level warnings when fallbacks were used'''

    _ensure_rdkit()
    if include_fingerprints is None:
        include_fingerprints = RDKIT_DEFAULT_INCLUDE_FINGERPRINTS
    if include_3d is None:
        include_3d = RDKIT_DEFAULT_INCLUDE_3D

    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    warnings: list[str] = []
    features: dict[str, FeatureValue] = {}

    # Physicochemical
    features["MolecularWeight"] = float(Descriptors.MolWt(mol))
    features["HeavyAtomMolWt"] = float(Descriptors.HeavyAtomMolWt(mol))
    features["LogP"] = float(Crippen.MolLogP(mol))
    features["MolMR"] = float(Crippen.MolMR(mol))
    features["LogS"] = float(esol_logS(mol))
    features["TPSA"] = float(rdMolDescriptors.CalcTPSA(mol))
    features["LabuteASA"] = float(rdMolDescriptors.CalcLabuteASA(mol))

    # Topology and complexity
    features["BalabanIndex"] = float(GraphDescriptors.BalabanJ(mol))
    features["Chi0"] = float(GraphDescriptors.Chi0(mol))
    features["Chi1"] = float(GraphDescriptors.Chi1(mol))
    features["Kappa1"] = float(GraphDescriptors.Kappa1(mol))
    features["Kappa2"] = float(GraphDescriptors.Kappa2(mol))
    features["Kappa3"] = float(GraphDescriptors.Kappa3(mol))
    
    # Use rdMolDescriptors.CalcPhi when available (Descriptors.Phi is not present in all builds)
    features["Phi"] = _safe_number(lambda: rdMolDescriptors.CalcPhi(mol), "Phi", warnings, cast=float)
    features["FractionCSP3"] = float(rdMolDescriptors.CalcFractionCSP3(mol))
    features["WienerIndex"] = float(_compute_wiener_index(mol))
    features["BertzCT"] = float(Descriptors.BertzCT(mol))
    if "nSPS" in RDKIT_FEATURES:
        features["nSPS"] = _compute_n_sps(mol, warnings)

    # Electronic/E-state descriptors
    features["MaxEStateIndex"] = _safe_number(lambda: Descriptors.MaxEStateIndex(mol), "MaxEStateIndex", warnings, cast=float)
    features["MinEStateIndex"] = _safe_number(lambda: Descriptors.MinEStateIndex(mol), "MinEStateIndex", warnings, cast=float)
    features["MaxAbsEStateIndex"] = _safe_number(lambda: Descriptors.MaxAbsEStateIndex(mol), "MaxAbsEStateIndex", warnings, cast=float)
    features["MinAbsEStateIndex"] = _safe_number(lambda: Descriptors.MinAbsEStateIndex(mol), "MinAbsEStateIndex", warnings, cast=float)

    # Partial charge descriptors
    charges, charges_warning = _compute_gasteiger_charges(mol)
    features["GasteigerCharges"] = charges
    if charges_warning:
        warnings.append(charges_warning)
    features["MaxPartialCharge"] = _safe_number(lambda: Descriptors.MaxPartialCharge(mol), "MaxPartialCharge", warnings, cast=float)
    features["MinPartialCharge"] = _safe_number(lambda: Descriptors.MinPartialCharge(mol), "MinPartialCharge", warnings, cast=float)
    features["MaxAbsPartialCharge"] = _safe_number(lambda: Descriptors.MaxAbsPartialCharge(mol), "MaxAbsPartialCharge", warnings, cast=float)
    features["MinAbsPartialCharge"] = _safe_number(lambda: Descriptors.MinAbsPartialCharge(mol), "MinAbsPartialCharge", warnings, cast=float)

    # Atom/bond counts
    heavy_count = _safe_number(lambda: Descriptors.HeavyAtomCount(mol), "HeavyAtomCount", warnings, cast=int)
    num_hetero = _safe_number(lambda: rdMolDescriptors.CalcNumHeteroatoms(mol), "NumHeteroAtoms", warnings, cast=int)
    num_radical = _safe_number(lambda: Descriptors.NumRadicalElectrons(mol), "NumRadicalElectrons", warnings, cast=int)
    features["HeavyAtomCount"] = heavy_count
    features["NumValenceElectrons"] = _safe_number(lambda: Descriptors.NumValenceElectrons(mol), "NumValenceElectrons", warnings, cast=float)
    features["NumRadicals"] = num_radical
    features["NumRadicalElectrons"] = num_radical
    features["NumHDonors"] = int(Lipinski.NumHDonors(mol))
    features["NumHAcceptors"] = int(Lipinski.NumHAcceptors(mol))
    features["NumRotatableBonds"] = int(Lipinski.NumRotatableBonds(mol))
    features["NumHeteroAtoms"] = num_hetero
    features["NHOHCount"] = int(Lipinski.NHOHCount(mol))
    features["NOCount"] = int(Lipinski.NOCount(mol))
    features["NumAmideBonds"] = _safe_number(lambda: rdMolDescriptors.CalcNumAmideBonds(mol), "NumAmideBonds", warnings, cast=int)

    # Ring and scaffold breakdown
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

    # Fingerprint density and quality metrics
    features["FpDensityMorgan1"] = _safe_number(lambda: Descriptors.FpDensityMorgan1(mol), "FpDensityMorgan1", warnings, cast=float)
    features["QED"] = float(QED.qed(mol))
    features["LipinskiRO5"] = _lipinski_ro5_pass(mol)

    # Van der Waals volume
    vdw_volume, vdw_warning = calc_vdw_volume(mol)
    features["VanDerWaalsVolume"] = float(vdw_volume)
    if vdw_warning:
        warnings.append(vdw_warning)

    # Optional 3D dipole
    if include_3d:
        dipole, dipole_warning = _compute_dipole_moment(mol)
        if dipole is None:
            features["DipoleMoment"] = float("nan")
            if dipole_warning:
                warnings.append(dipole_warning)
        else:
            features["DipoleMoment"] = float(dipole)
    else:
        features["DipoleMoment"] = float("nan")

    # Optional fingerprints
    if include_fingerprints:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        features["ECFP4"] = [int(x) for x in fp.ToBitString()]
        features["MACCS"] = _compute_maccs_bits(mol, warnings)
    else:
        features["ECFP4"] = []
        features["MACCS"] = []

    missing = [k for k in RDKIT_FEATURES if k not in features]
    if missing:
        raise ValueError(f"Missing computed features: {missing}")

    return features, warnings


def compute_formal_charge(smiles: str) -> int:
    '''Compute formal charge for a SMILES string'''
    mol = mol_from_smiles(smiles)
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def graph_diameter(smiles: str) -> float:
    '''Compute graph diameter as the max shortest-path distance between atoms'''
    _ensure_rdkit()
    mol = mol_from_smiles(smiles)
    dist = Chem.GetDistanceMatrix(mol)
    return float(dist.max()) if dist.size > 0 else 0.0


def _k_hop_env_atoms(mol: "Chem.Mol", center_idx: int, hops: int = 3) -> list[int]:
    '''Return atom indices in a k-hop neighborhood using breadth-first expansion'''
    seen = {center_idx}
    frontier = {center_idx}
    for _ in range(hops):
        nxt: set[int] = set()
        for atom_idx in frontier:
            atom = mol.GetAtomWithIdx(atom_idx)
            nxt.update(nei.GetIdx() for nei in atom.GetNeighbors())
        frontier = nxt - seen
        seen |= frontier
    return list(seen)


def anchor_to_anchor_graph_distance(
    smiles: str,
    anchor_symbol: str = "Lr",
    bond_len: float = 1.45,
) -> float:
    '''
    Compute a graph-based anchor-to-anchor distance proxy (Angstrom-like units).
    Returns NaN when fewer than two anchor atoms are present'''

    _ensure_rdkit()
    mol = mol_from_smiles(smiles)
    anchors = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == anchor_symbol]
    if len(anchors) < 2: return float("nan")

    max_d = 0.0
    for i, j in combinations(anchors, 2):
        path = Chem.GetShortestPath(mol, i, j)
        bonds = max(0, len(path) - 1)
        max_d = max(max_d, float(bonds) * bond_len)
    return max_d


def anchor_env_symmetry_proxy(
    smiles: str,
    env_hops: int = 4,
    fp_radius: int = 2,
    n_bits: int = 2048,
    agg: str = "softmin",
) -> float:
    '''
    Estimate linker symmetry from similarity of local environments around [Lr] anchors
    Returns NaN when anchor-based symmetry cannot be computed'''

    _ensure_rdkit()
    if DataStructs is None: return float("nan")

    mol = mol_from_smiles(smiles)
    anchors = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Lr"]
    if len(anchors) < 2: return float("nan")

    fps = []
    for idx in anchors:
        atoms = _k_hop_env_atoms(mol, idx, hops=env_hops)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius = fp_radius,
            nBits = n_bits,
            fromAtoms = atoms,
        )
        fps.append(fp)

    sims = [
        float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        for i in range(len(fps))
        for j in range(i + 1, len(fps))
    ]
    if not sims:
        return float("nan")

    mean_s = float(sum(sims) / len(sims))
    if agg == "mean": return max(0.0, min(1.0, mean_s))
    if agg == "min": return max(0.0, min(1.0, min(sims)))

    # softmin: mean - std penalizes asymmetric pairs while remaining smooth.
    var = sum((s - mean_s) ** 2 for s in sims) / len(sims)
    softmin = mean_s - math.sqrt(var)
    return max(0.0, min(1.0, softmin))


def topological_symmetry_proxy(smiles: str) -> float:
    ''' Compute a generic symmetry proxy from canonical atom-rank multiplicity
    0.0 -> mostly unique atom environments
    1.0 -> many repeated equivalent environments'''

    _ensure_rdkit()
    mol = mol_from_smiles(smiles)
    heavy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy_indices:
        return float("nan")

    ranks = Chem.CanonicalRankAtoms(mol, breakTies = False)
    counts: dict[int, int] = {}
    for idx in heavy_indices:
        rank = int(ranks[idx])
        counts[rank] = counts.get(rank, 0) + 1

    repeated = sum(cnt for cnt in counts.values() if cnt > 1)
    return max(0.0, min(1.0, float(repeated) / float(len(heavy_indices))))
