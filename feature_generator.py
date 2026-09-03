# =============================================================================
# feature_generator.py
# PART 1
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    GraphDescriptors,
    Crippen,
    rdMolDescriptors,
    Descriptors3D
)

# =============================================================================
# SAFE DIVISION
# =============================================================================

def safe_div(a, b):

    if pd.isna(a):
        return np.nan

    if pd.isna(b):
        return np.nan

    if abs(b) < 1e-12:
        return 0.0

    return a / b


# =============================================================================
# TOPOLOGICAL INDICES
# =============================================================================

def wiener_index(mol):

    dmat = Chem.GetDistanceMatrix(mol)

    n = dmat.shape[0]

    total = 0.0

    for i in range(n):

        for j in range(i + 1, n):

            total += dmat[i, j]

    return float(total)


def zagreb_index(mol):

    total = 0.0

    for atom in mol.GetAtoms():

        total += atom.GetDegree() ** 2

    return float(total)


def randic_index(mol):

    value = 0.0

    for bond in mol.GetBonds():

        d1 = bond.GetBeginAtom().GetDegree()

        d2 = bond.GetEndAtom().GetDegree()

        if d1 > 0 and d2 > 0:

            value += 1.0 / np.sqrt(d1 * d2)

    return float(value)


def petitjean_index(mol):

    dmat = Chem.GetDistanceMatrix(mol)

    ecc = dmat.max(axis=1)

    diameter = ecc.max()

    radius = ecc.min()

    if diameter == 0:

        return 0.0

    return (diameter - radius) / diameter


# =============================================================================
# SMARTS PATTERNS
# =============================================================================

FG_SMARTS = {

    "Carboxylic acid":"[CX3](=O)[OX2H1]",

    "Amide":"[NX3][CX3](=O)",

    "Ester":"[CX3](=O)[OX2][#6]",

    "Alcohol":"[OX2H][CX4]",

    "Phenol":"c[OX2H]",

    "Primary amine":"[NX3;H2]",

    "Secondary amine":"[NX3;H1]",

    "Tertiary amine":"[NX3;H0]",

    "Ketone":"[#6][CX3](=O)[#6]",

    "Aldehyde":"[CX3H1](=O)",

    "Ether":"[OD2]([#6])[#6]",

    "Nitro":"[NX3](=O)=O",

    "Nitrile":"C#N",

    "Sulfoxide":"S(=O)",

    "Sulfone":"S(=O)(=O)",

    "Thiol":"[SX2H]",

    "Phosphate":"P(=O)(O)(O)",

    "Halide":"[F,Cl,Br,I]"
}


# =============================================================================
# DOMINANT FUNCTIONAL GROUP
# =============================================================================

def detect_dominant_functional_group(mol):

    for name, smarts in FG_SMARTS.items():

        patt = Chem.MolFromSmarts(smarts)

        if mol.HasSubstructMatch(patt):

            return name

    return "Hydrocarbon"


# =============================================================================
# HYDROGEN BOND CLASS
# =============================================================================

def hydrogen_bond_class(mol):

    hbd = rdMolDescriptors.CalcNumHBD(mol)

    hba = rdMolDescriptors.CalcNumHBA(mol)

    if hbd > 0 and hba > 0:

        return "Donor & Acceptor"

    elif hbd > 0:

        return "Donor"

    elif hba > 0:

        return "Acceptor"

    else:

        return "Non H-bonding"


# =============================================================================
# POLARITY CLASS
# =============================================================================

def polarity_class(logp, tpsa):

    if tpsa < 20:

        return "Nonpolar"

    elif tpsa < 40:

        return "Weakly polar"

    elif tpsa < 75:

        return "Moderately polar"

    else:

        return "Highly polar"


# =============================================================================
# MOLECULAR FAMILY
# =============================================================================

def molecular_family(hbd, hba):

    if hbd > 0:

        return "Polar protic"

    elif hba > 0:

        return "Polar aprotic"

    else:

        return "Nonpolar"


# =============================================================================
# GENERATE 3D MOLECULE
# =============================================================================

def generate_3d_mol(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()

    params.randomSeed = 42

    status = AllChem.EmbedMolecule(
        mol,
        params
    )

    if status != 0:

        status = AllChem.EmbedMolecule(
            mol,
            useRandomCoords=True
        )

    if status != 0:

        return None

    try:

        if AllChem.MMFFHasAllMoleculeParams(mol):

            AllChem.MMFFOptimizeMolecule(mol)

        else:

            AllChem.UFFOptimizeMolecule(mol)

    except:

        pass

    return mol


# =============================================================================
# END OF PART 1
# =============================================================================
# =============================================================================
# PART 2
# SOLVATION DESCRIPTORS (2D + GRAPH DESCRIPTORS)
# =============================================================================

def solvation_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # ----------------------------------------------------------
    # Basic descriptors
    # ----------------------------------------------------------

    logp = Crippen.MolLogP(mol)

    tpsa = rdMolDescriptors.CalcTPSA(mol)

    hbd = rdMolDescriptors.CalcNumHBD(mol)

    hba = rdMolDescriptors.CalcNumHBA(mol)

    desc = {}

    # ==========================================================
    # BASIC DESCRIPTORS
    # ==========================================================

    desc["MolWt"] = Descriptors.MolWt(mol)

    desc["ExactMolWt"] = Descriptors.ExactMolWt(mol)

    desc["LogP"] = logp

    desc["TPSA"] = tpsa

    desc["MolMR"] = Crippen.MolMR(mol)

    desc["HeavyAtomCount"] = mol.GetNumHeavyAtoms()

    desc["NumRotatableBonds"] = \
        rdMolDescriptors.CalcNumRotatableBonds(mol)

    desc["NumRings"] = \
        rdMolDescriptors.CalcNumRings(mol)

    desc["NumAromaticRings"] = \
        rdMolDescriptors.CalcNumAromaticRings(mol)

    desc["FracCSP3"] = \
        rdMolDescriptors.CalcFractionCSP3(mol)

    desc["FormalCharge"] = \
        Chem.GetFormalCharge(mol)

    # ==========================================================
    # HYDROGEN BOND DESCRIPTORS
    # ==========================================================

    desc["HBD"] = hbd

    desc["HBA"] = hba

    desc["HydrogenBondClass"] = \
        hydrogen_bond_class(mol)

    desc["DominantFunctionalGroup"] = \
        detect_dominant_functional_group(mol)

    desc["Family"] = \
        molecular_family(hbd, hba)

    desc["PolarityClass"] = \
        polarity_class(logp, tpsa)

    # ==========================================================
    # GRAPH DESCRIPTORS
    # ==========================================================

    try:
        desc["BalabanJ"] = \
            GraphDescriptors.BalabanJ(mol)
    except:
        desc["BalabanJ"] = np.nan

    try:
        desc["BertzCT"] = \
            GraphDescriptors.BertzCT(mol)
    except:
        desc["BertzCT"] = np.nan

    try:
        desc["Chi0"] = \
            GraphDescriptors.Chi0(mol)
    except:
        desc["Chi0"] = np.nan

    try:
        desc["Chi1"] = \
            GraphDescriptors.Chi1(mol)
    except:
        desc["Chi1"] = np.nan

    try:
        desc["HallKierAlpha"] = \
            GraphDescriptors.HallKierAlpha(mol)
    except:
        desc["HallKierAlpha"] = np.nan

    try:
        desc["Kappa1"] = \
            GraphDescriptors.Kappa1(mol)
    except:
        desc["Kappa1"] = np.nan

    try:
        desc["Kappa2"] = \
            GraphDescriptors.Kappa2(mol)
    except:
        desc["Kappa2"] = np.nan

    # ==========================================================
    # TOPOLOGICAL INDICES
    # ==========================================================

    try:
        desc["WienerIndex"] = \
            wiener_index(mol)
    except:
        desc["WienerIndex"] = np.nan

    try:
        desc["ZagrebIndex"] = \
            zagreb_index(mol)
    except:
        desc["ZagrebIndex"] = np.nan

    try:
        desc["RandicIndex"] = \
            randic_index(mol)
    except:
        desc["RandicIndex"] = np.nan

    try:
        desc["PetitjeanIndex"] = \
            petitjean_index(mol)
    except:
        desc["PetitjeanIndex"] = np.nan

    # ==========================================================
    # GENERATE 3D MOLECULE
    # ==========================================================

    mol3d = generate_3d_mol(smiles)

    if mol3d is None:

        desc["MolVolume"] = np.nan
        desc["LabuteASA"] = np.nan
        desc["RadiusOfGyration"] = np.nan
        desc["Asphericity"] = np.nan
        desc["Eccentricity"] = np.nan
        desc["InertialShapeFactor"] = np.nan
        desc["SpherocityIndex"] = np.nan
        desc["PMI1"] = np.nan
        desc["PMI2"] = np.nan
        desc["PMI3"] = np.nan
        desc["PMI_ratio_1_2"] = np.nan
        desc["PMI_ratio_2_3"] = np.nan

        return desc

    # ==========================================================
    # PART 3 CONTINUES HERE...
    # ==========================================================

    # ==========================================================
    # PART 3
    # 3D DESCRIPTORS
    # ==========================================================

    # ----------------------------------------------------------
    # Molecular Volume
    # ----------------------------------------------------------

    try:
        desc["MolVolume"] = AllChem.ComputeMolVolume(mol3d)
    except:
        desc["MolVolume"] = np.nan

    # ----------------------------------------------------------
    # Labute Surface Area
    # ----------------------------------------------------------

    try:
        desc["LabuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)
    except:
        desc["LabuteASA"] = np.nan

    # ----------------------------------------------------------
    # Radius Of Gyration
    # ----------------------------------------------------------

    try:
        desc["RadiusOfGyration"] = \
            Descriptors3D.RadiusOfGyration(mol3d)
    except:
        desc["RadiusOfGyration"] = np.nan

    # ----------------------------------------------------------
    # Asphericity
    # ----------------------------------------------------------

    try:
        desc["Asphericity"] = \
            Descriptors3D.Asphericity(mol3d)
    except:
        desc["Asphericity"] = np.nan

    # ----------------------------------------------------------
    # Eccentricity
    # ----------------------------------------------------------

    try:
        desc["Eccentricity"] = \
            Descriptors3D.Eccentricity(mol3d)
    except:
        desc["Eccentricity"] = np.nan

    # ----------------------------------------------------------
    # Inertial Shape Factor
    # ----------------------------------------------------------

    try:
        desc["InertialShapeFactor"] = \
            Descriptors3D.InertialShapeFactor(mol3d)
    except:
        desc["InertialShapeFactor"] = np.nan

    # ----------------------------------------------------------
    # Spherocity Index
    # ----------------------------------------------------------

    try:
        desc["SpherocityIndex"] = \
            Descriptors3D.SpherocityIndex(mol3d)
    except:
        desc["SpherocityIndex"] = np.nan

    # ----------------------------------------------------------
    # Principal Moments of Inertia
    # ----------------------------------------------------------

    try:
        desc["PMI1"] = Descriptors3D.PMI1(mol3d)
    except:
        desc["PMI1"] = np.nan

    try:
        desc["PMI2"] = Descriptors3D.PMI2(mol3d)
    except:
        desc["PMI2"] = np.nan

    try:
        desc["PMI3"] = Descriptors3D.PMI3(mol3d)
    except:
        desc["PMI3"] = np.nan

    # ----------------------------------------------------------
    # PMI Ratios
    # ----------------------------------------------------------

    desc["PMI_ratio_1_2"] = safe_div(
        desc["PMI1"],
        desc["PMI2"]
    )

    desc["PMI_ratio_2_3"] = safe_div(
        desc["PMI2"],
        desc["PMI3"]
    )

    # ==========================================================
    # VERIFY ALL NUMERIC VALUES
    # ==========================================================

    for key, value in desc.items():

        if isinstance(value, (float, int, np.number)):

            if np.isinf(value):
                desc[key] = np.nan

    return desc
# =============================================================================
# PART 4
# BUILD FEATURE VECTOR
# =============================================================================

def build_features(solute_smiles, solvent_smiles):

    # ---------------------------------------------------------
    # Generate descriptors
    # ---------------------------------------------------------

    solute = solvation_descriptors(solute_smiles)

    solvent = solvation_descriptors(solvent_smiles)

    if solute is None:
        return None

    if solvent is None:
        return None

    features = {}

    # ---------------------------------------------------------
    # Original smiles
    # ---------------------------------------------------------

    features["mol_solute"] = solute_smiles

    features["mol_solvent"] = solvent_smiles

    # ---------------------------------------------------------
    # Solute descriptors
    # ---------------------------------------------------------

    for key, value in solute.items():

        features[f"solute_{key}"] = value

    # ---------------------------------------------------------
    # Solvent descriptors
    # ---------------------------------------------------------

    for key, value in solvent.items():

        features[f"solvent_{key}"] = value

    # ---------------------------------------------------------
    # Numeric descriptor names
    # ---------------------------------------------------------

    # ---------------------------------------------------------
# Numeric descriptor names used during TRAINING
# ---------------------------------------------------------

    numeric_keys = [

        "MolWt",
        "ExactMolWt",
        "LogP",
        "TPSA",
        "MolMR",
        "HeavyAtomCount",
        "NumRotatableBonds",
        "NumRings",
        "NumAromaticRings",
        "FracCSP3",

        "HBD",
        "HBA",

        "BalabanJ",
        "BertzCT",
        "Chi0",
        "Chi1",
        "HallKierAlpha",
        "Kappa1",
        "Kappa2",

        "WienerIndex",
        "ZagrebIndex",
        "RandicIndex",
        "PetitjeanIndex",

        "MolVolume",
        "LabuteASA",
        "RadiusOfGyration",

        "PMI1",
        "PMI2",
        "PMI3"
    ]

    # ---------------------------------------------------------
    # Interaction Features
    # ---------------------------------------------------------

    for key in numeric_keys:

        if key not in solvent:
            continue

        try:

            s = float(solute[key])

            v = float(solvent[key])

        except:

            continue

        # Difference

        features[f"diff_{key}"] = abs(s - v)

        # Product

        features[f"prod_{key}"] = s * v

        # Ratio

        features[f"ratio_{key}"] = safe_div(s, v)

    # ---------------------------------------------------------
    # Replace inf
    # ---------------------------------------------------------

    for key in features:

        value = features[key]

        if isinstance(value, (int, float, np.integer, np.floating)):

            if np.isinf(value):

                features[key] = np.nan

    return features
# =============================================================================
# PART 5
# GENERATE COMPLETE FEATURE DATASET
# =============================================================================

from tqdm import tqdm

tqdm.pandas()


def generate_dataset(
    df,
    solute_col="mol_solute",
    solvent_col="mol_solvent",
    target_col="target"
):

    rows = []

    failed = []

    total = len(df)

    print("=" * 70)
    print("Generating RDKit Features")
    print("=" * 70)

    for idx, row in tqdm(
        df.iterrows(),
        total=total
    ):

        try:

            solute_smiles = str(row[solute_col]).strip()

            solvent_smiles = str(row[solvent_col]).strip()

            target = row[target_col]

            feature_row = build_features(
                solute_smiles,
                solvent_smiles
            )

            if feature_row is None:

                failed.append(idx)

                continue

            feature_row["target"] = target

            rows.append(feature_row)

        except Exception as e:

            print(
                f"Failed Row {idx}: {e}"
            )

            failed.append(idx)

    feature_df = pd.DataFrame(rows)

    # =====================================================
    # Move target to third column
    # =====================================================

    cols = feature_df.columns.tolist()

    if "target" in cols:

        cols.remove("target")

        cols.insert(2, "target")

        feature_df = feature_df[cols]

    print()

    print("=" * 70)

    print("Finished")

    print("=" * 70)

    print("Generated Samples :", len(feature_df))

    print("Failed Samples    :", len(failed))

    print("=" * 70)

    return feature_df, failed


# =============================================================================
# SAVE FEATURE DATASET
# =============================================================================

def save_dataset(
    feature_df,
    output_csv="generated_featuresNEW.csv"
):

    feature_df.to_csv(
        output_csv,
        index=False
    )

    print()

    print("=" * 70)

    print("Saved Successfully")

    print(output_csv)

    print("=" * 70)


# =============================================================================
# NULL REPORT
# =============================================================================

def null_report(feature_df):

    report = pd.DataFrame({

        "Column": feature_df.columns,

        "Null_Count":
        feature_df.isnull().sum().values

    })

    report = report.sort_values(

        "Null_Count",

        ascending=False

    )

    return report


# =============================================================================
# COLUMN REPORT
# =============================================================================

def column_report(feature_df):

    report = pd.DataFrame({

        "No": range(

            1,

            len(feature_df.columns) + 1

        ),

        "Column": feature_df.columns

    })

    return report


# =============================================================================
# FAILED ROWS
# =============================================================================

def failed_rows(df, failed):

    return df.iloc[failed].copy()


# =============================================================================
# PART 6 STARTS HERE
# =============================================================================
# =============================================================================
# PART 6
# MAIN PROGRAM
# =============================================================================

def detect_columns(df):

    # ----------------------------
    # Solute column
    # ----------------------------

    solute_candidates = [

        "mol_solute",
        "Solute_SMILES",
        "smiles_solute",
        "solute_smiles",
        "solute"
    ]

    solvent_candidates = [

        "mol_solvent",
        "Solvent_SMILES",
        "smiles_solvent",
        "solvent_smiles",
        "solvent"
    ]

    target_candidates = [

        "target",
        "DeltaGsolv",
        "dGsolv_avg",
        "DeltaG",
        "deltaG"
    ]

    solute_col = None
    solvent_col = None
    target_col = None

    for c in solute_candidates:

        if c in df.columns:

            solute_col = c

            break

    for c in solvent_candidates:

        if c in df.columns:

            solvent_col = c

            break

    for c in target_candidates:

        if c in df.columns:

            target_col = c

            break

    if solute_col is None:

        raise ValueError(
            "Solute column not found."
        )

    if solvent_col is None:

        raise ValueError(
            "Solvent column not found."
        )

    if target_col is None:

        raise ValueError(
            "Target column not found."
        )

    return (

        solute_col,

        solvent_col,

        target_col

    )


# =============================================================================
# RUN FEATURE GENERATOR
# =============================================================================

def run_feature_generator(

    input_csv,

    output_csv="RDKit_Features.csv"

):

    print()

    print("=" * 70)

    print("Loading Dataset")

    print("=" * 70)

    df = pd.read_csv(input_csv)

    print(df.shape)

    print()

    solute_col, solvent_col, target_col = detect_columns(df)

    print("Detected Columns")

    print()

    print("Solute :", solute_col)

    print("Solvent:", solvent_col)

    print("Target :", target_col)

    print()

    feature_df, failed = generate_dataset(

        df,

        solute_col,

        solvent_col,

        target_col

    )

    save_dataset(

        feature_df,

        output_csv

    )

    print()

    print("=" * 70)

    print("NULL REPORT")

    print("=" * 70)

    print(

        null_report(feature_df)

    )

    print()

    print("=" * 70)

    print("COLUMN REPORT")

    print("=" * 70)

    print(

        column_report(feature_df)

    )

    if len(failed) > 0:

        failed_df = failed_rows(

            df,

            failed

        )

        failed_df.to_csv(

            "failed_rows.csv",

            index=False

        )

        print()

        print(

            "Failed rows saved as "

            "failed_rows.csv"

        )

    print()

    print("=" * 70)

    print("Finished Successfully")

    print("=" * 70)

    return feature_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    INPUT_FILE = "/home/ysws/Mahesh/MnSol_Combisol_merge/combined_MNsol_CombiSolNEW.csv"

    OUTPUT_FILE = "combined_MNsol_CombiSol_RDKit_188_desc.csv"

    feature_df = run_feature_generator(

        INPUT_FILE,

        OUTPUT_FILE

    )