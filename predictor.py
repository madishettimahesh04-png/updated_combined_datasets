import os
import joblib
import torch
import pandas as pd

from torch_geometric.utils import from_smiles
from torch_geometric.data import Batch

from model import Model
from feature_generator import build_features

# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = os.path.join(BASE_DIR, "models")

# --------------------------------------------------
# Load Saved Objects
# --------------------------------------------------

pipeline = joblib.load(
    os.path.join(MODELS, "pipeline.pkl")
)

config = joblib.load(
    os.path.join(MODELS, "model_config.pkl")
)

feature_order = joblib.load(
    os.path.join(MODELS, "feature_order.pkl")
)

variance_input_columns = joblib.load(
    os.path.join(MODELS, "variance_input_columns.pkl")
)

print("=" * 60)
print("Pipeline Loaded")
print("=" * 60)

print("Pipeline Keys:")
print(pipeline.keys())

print("\nOHE Columns :", len(pipeline["ohe_columns"]))
print("Selected Columns :", len(pipeline["selected_columns"]))
print("Removed Corr :", len(pipeline["removed_corr_features"]))
print("Final Features :", len(feature_order))
# ==========================================================
# Load Hybrid GNN Model
# ==========================================================

print("\n" + "=" * 60)
print("Loading Hybrid GNN Model")
print("=" * 60)

model = Model(
    desc_dim=config["input_dim"],
    hidden_dim=config["hidden_dim"],
    heads=config["heads"],
    dropout=config["dropout"],
    desc_hidden=config["desc_hidden"],
    mlp_hidden=config["mlp_hidden"]
).to(DEVICE)

# --------------------------------------------------
# Load trained weights
# --------------------------------------------------

checkpoint = torch.load(
    os.path.join(MODELS, "best_model.pt"),
    map_location=DEVICE,
    weights_only=False
)

# Handle both checkpoint formats
if isinstance(checkpoint, dict):

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model.eval()

print("✓ Model loaded successfully")
# ==========================================================
# PART 3
# SMILES → GRAPH CONVERSION
# ==========================================================

from torch_geometric.utils import from_smiles
from torch_geometric.data import Batch


def smiles_to_graph(smiles):
    """
    Convert a SMILES string into a PyTorch Geometric graph.
    """

    try:

        graph = from_smiles(smiles)

        # Convert node features to float
        graph.x = graph.x.float()

        # Convert edge features to float
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr.float()

        return graph

    except Exception as e:

        raise ValueError(
            f"Unable to generate graph for SMILES:\n{smiles}\n\n{e}"
        )

# ==========================================================
# Descriptor Preparation
# ==========================================================

def prepare_descriptor(solute_smiles, solvent_smiles):
    """
    Generate descriptors exactly as used during training.
    """

    # --------------------------------------------------
    # Generate descriptors
    # --------------------------------------------------

    df = build_features(solute_smiles, solvent_smiles)

    if df is None:
        raise ValueError("Feature generation failed.")

    if isinstance(df, dict):
        df = pd.DataFrame([df])

    # --------------------------------------------------
    # One-Hot Encoding
    # --------------------------------------------------

    cat_cols = [
        "solute_HydrogenBondClass",
        "solute_DominantFunctionalGroup",
        "solute_Family",
        "solute_PolarityClass",
        "solvent_HydrogenBondClass",
        "solvent_DominantFunctionalGroup",
        "solvent_Family",
        "solvent_PolarityClass",
    ]

    df = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True
    )

    # --------------------------------------------------
    # Match EXACT columns used during VarianceThreshold fit
    # --------------------------------------------------

    # Remove non-feature columns
    training_columns = [
        c for c in pipeline["ohe_columns"]
        if c not in ["mol_solute", "mol_solvent", "target"]
    ]

    # Remove columns that were never used during training
    df = df[[c for c in df.columns if c in training_columns]]

    # Add missing columns
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    # Exact training order
    df = df.reindex(columns=training_columns)

    # --------------------------------------------------
    # Match EXACT VarianceThreshold input
    # --------------------------------------------------

    fit_cols = list(pipeline["variance_selector"].feature_names_in_)

    df = df[[c for c in df.columns if c in fit_cols]]

    for col in fit_cols:
        if col not in df.columns:
            df[col] = 0

    df = df.reindex(columns=fit_cols)

    # --------------------------------------------------
    # Variance Threshold
    # 230 -> 163
    # --------------------------------------------------

    df = pipeline["variance_selector"].transform(df)

    df = pd.DataFrame(
        df,
        columns=pipeline["selected_columns"]
    )

    # --------------------------------------------------
    # Remove correlated features
    # 163 -> 123
    # --------------------------------------------------

    df = df.drop(
        columns=pipeline["removed_corr_features"],
        errors="ignore"
    )

    # --------------------------------------------------
    # Final feature order
    # --------------------------------------------------

    df = df.reindex(
        columns=feature_order,
        fill_value=0
    )

    # --------------------------------------------------
    # Scale descriptors
    # --------------------------------------------------

    df = pipeline["scaler"].transform(df)

    return df

# ==========================================================
# PART 5
# SINGLE PREDICTION
# ==========================================================

@torch.no_grad()
def predict(solute_smiles, solvent_smiles):
    """
    Predict solvation free energy (ΔG)
    from a solute-solvent SMILES pair.
    """

    # --------------------------------------------------
    # Generate descriptor vector
    # --------------------------------------------------

    descriptor = prepare_descriptor(
        solute_smiles,
        solvent_smiles
    )

    descriptor = torch.tensor(
        descriptor,
        dtype=torch.float32,
        device=DEVICE
    )

    # --------------------------------------------------
    # Generate molecular graphs
    # --------------------------------------------------

    g_solute = smiles_to_graph(solute_smiles)
    g_solvent = smiles_to_graph(solvent_smiles)

    g_solute = Batch.from_data_list([g_solute]).to(DEVICE)
    g_solvent = Batch.from_data_list([g_solvent]).to(DEVICE)

    # --------------------------------------------------
    # Model prediction
    # --------------------------------------------------

    prediction = model(
        g_solute,
        g_solvent,
        descriptor
    )

    # --------------------------------------------------
    # Convert tensor to float
    # --------------------------------------------------

    prediction = prediction.squeeze()

    return float(prediction.cpu().item())
# ==========================================================
# PART 6
# BATCH CSV PREDICTION
# ==========================================================

from tqdm import tqdm


def predict_csv(
    input_csv,
    output_csv="predictions.csv"
):
    """
    Predict ΔG for every row in a CSV file.
    """

    # --------------------------------------------------
    # Read CSV
    # --------------------------------------------------

    df = pd.read_csv(input_csv)

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    print(f"Rows : {len(df)}")
    print(f"Columns : {list(df.columns)}")

    # --------------------------------------------------
    # Validate Required Columns
    # --------------------------------------------------

    required_columns = [
        "Solute SMILES",
        "Solvent SMILES"
    ]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(
                f"Missing required column: {col}"
            )

    # --------------------------------------------------
    # Prediction Loop
    # --------------------------------------------------

    predictions = []

    tqdm.pandas()

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Predicting"
    ):

        try:

            pred = predict(
                row["Solute SMILES"],
                row["Solvent SMILES"]
            )

        except Exception as e:

            print(f"\nRow {idx} failed")
            print(e)

            pred = None

        predictions.append(pred)

    # --------------------------------------------------
    # Save Predictions
    # --------------------------------------------------

    df["Predicted_dG"] = predictions

    df.to_csv(
        output_csv,
        index=False
    )

    print("\n" + "=" * 60)
    print("Prediction Completed")
    print("=" * 60)

    print(f"Input File : {input_csv}")
    print(f"Output File: {output_csv}")
    print(f"Rows Saved : {len(df)}")

    return output_csv

if __name__ == "__main__":

    print("Testing predictor...")

    pred = predict(
        "CCO",   # ethanol
        "O"      # water
    )

    print(pred)