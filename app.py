import streamlit as st
from rdkit import Chem
from predictor import predict


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hybrid GNN ΔG Predictor",
    page_icon="🧪",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧪 Hybrid GNN Solvation Free Energy Predictor")

st.write(
    "Predict the solvation free energy (ΔG) of a solute in a solvent "
    "using the trained Hybrid GNN model."
)

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
    "🔮 Predict ΔG",
    type="primary",
    use_container_width=True
):

    solute = solute_smiles.strip()
    solvent = solvent_smiles.strip()

    # Check empty inputs
    if not solute or not solvent:
        st.warning("Please enter both solute and solvent SMILES.")

    else:

        # Validate solute
        solute_mol = Chem.MolFromSmiles(solute)

        if solute_mol is None:
            st.error("❌ Invalid solute SMILES.")
            st.stop()

        # Validate solvent
        solvent_mol = Chem.MolFromSmiles(solvent)

        if solvent_mol is None:
            st.error("❌ Invalid solvent SMILES.")
            st.stop()

        # Run prediction
        with st.spinner("Running Hybrid GNN prediction..."):

            try:

                prediction = predict(
                    solute,
                    solvent
                )

                st.success("Prediction completed successfully.")

                st.metric(
                    label="Predicted Solvation Free Energy (ΔG)",
                    value=f"{prediction:.4f} kcal mol⁻¹"
                )

            except Exception as e:

                st.error("Prediction failed.")

                st.exception(e)


# ============================================================
# INFORMATION
# ============================================================

with st.expander("Example"):

    st.write("**Solute:** CCO")
    st.write("**Solvent:** O")

with st.expander("About the model"):

    st.write(
        "This application uses the trained Hybrid GNN model to predict "
        "solvation free energy from solute and solvent SMILES."
    )
