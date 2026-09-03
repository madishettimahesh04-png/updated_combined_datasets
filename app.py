import streamlit as st
from rdkit import Chem

from predictor import predict


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solvation Free Energy Predictor",
    page_icon="🧪",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧪 Solvation Free Energy Predictor")


st.divider()


# ============================================================
# INPUTS
# ============================================================

solute_smiles = st.text_input(
    "Solute SMILES",
    placeholder="Example: CCO"
)

solvent_smiles = st.text_input(
    "Solvent SMILES",
    placeholder="Example: O"
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict ΔG",
    type="primary",
    use_container_width=True
):

    solute = solute_smiles.strip()
    solvent = solvent_smiles.strip()

    # Check empty inputs
    if not solute:
        st.warning("Please enter the solute SMILES.")
        st.stop()

    if not solvent:
        st.warning("Please enter the solvent SMILES.")
        st.stop()

    # Validate solute SMILES
    if Chem.MolFromSmiles(solute) is None:
        st.error("Invalid solute SMILES.")
        st.stop()

    # Validate solvent SMILES
    if Chem.MolFromSmiles(solvent) is None:
        st.error("Invalid solvent SMILES.")
        st.stop()

    # Run prediction
    with st.spinner("Predicting ΔG..."):

        try:

            delta_g = predict(
                solute,
                solvent
            )

            delta_g = float(delta_g)

            st.success("Prediction completed.")

            st.metric(
                label="Predicted Solvation Free Energy (ΔG)",
                value=f"{delta_g:.4f} kcal mol⁻¹"
            )

        except Exception as e:

            st.error("Prediction failed.")
            st.exception(e)
