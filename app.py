import streamlit as st
from rdkit import Chem

from predictor import predict, config, feature_order


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hybrid GNN ΔG Predictor",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        text-align: center;
        margin-top: 20px;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 700;
    }

    .result-unit {
        font-size: 1rem;
        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🧪 Hybrid GNN Solvation Free Energy Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict solvation free energy (ΔG) from solute and solvent SMILES'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("Model Information")

# Number of input features
try:
    input_features = int(config["input_dim"])
except Exception:
    input_features = len(feature_order)

# Number of features in saved feature order
feature_count = len(feature_order)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Input Features",
        value=f"{input_features}"
    )

with col2:
    st.metric(
        label="Hidden Dimension",
        value=f"{config['hidden_dim']}"
    )

with col3:
    st.metric(
        label="Attention Heads",
        value=f"{config['heads']}"
    )


# ============================================================
# MODEL DETAILS
# ============================================================

with st.expander("Model Details"):

    st.write(
        f"**Input feature dimension:** {input_features}"
    )

    st.write(
        f"**Number of saved features:** {feature_count}"
    )

    st.write(
        f"**Hidden dimension:** {config['hidden_dim']}"
    )

    st.write(
        f"**Attention heads:** {config['heads']}"
    )

    st.write(
        f"**Dropout:** {config['dropout']}"
    )

    st.write(
        f"**Descriptor hidden dimension:** {config['desc_hidden']}"
    )

    st.write(
        f"**MLP hidden dimension:** {config['mlp_hidden']}"
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.divider()

st.subheader("Enter Molecular Structures")

st.write(
    "Enter valid SMILES representations for the solute and solvent."
)


# Solute input
solute_smiles = st.text_input(
    "Solute SMILES",
    placeholder="Example: CCO",
    help="Enter the SMILES string of the solute molecule."
)


# Solvent input
solvent_smiles = st.text_input(
    "Solvent SMILES",
    placeholder="Example: O",
    help="Enter the SMILES string of the solvent molecule."
)


# ============================================================
# EXAMPLES
# ============================================================

st.caption(
    "Example: Solute = CCO  |  Solvent = O"
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict ΔG",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    solute = solute_smiles.strip()
    solvent = solvent_smiles.strip()


    # --------------------------------------------------------
    # Check empty input
    # --------------------------------------------------------

    if not solute:

        st.warning(
            "Please enter a Solute SMILES."
        )

        st.stop()


    if not solvent:

        st.warning(
            "Please enter a Solvent SMILES."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate Solute SMILES
    # --------------------------------------------------------

    solute_mol = Chem.MolFromSmiles(solute)

    if solute_mol is None:

        st.error(
            "❌ Invalid Solute SMILES. "
            "Please enter a valid molecular SMILES string."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate Solvent SMILES
    # --------------------------------------------------------

    solvent_mol = Chem.MolFromSmiles(solvent)

    if solvent_mol is None:

        st.error(
            "❌ Invalid Solvent SMILES. "
            "Please enter a valid molecular SMILES string."
        )

        st.stop()


    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    with st.spinner(
        "Running Hybrid GNN prediction..."
    ):

        try:

            prediction = predict(
                solute,
                solvent
            )

            # Convert to float
            prediction = float(prediction)


            # ------------------------------------------------
            # Prediction result
            # ------------------------------------------------

            st.success(
                "Prediction completed successfully."
            )

            st.markdown(
                f"""
                <div class="result-box">

                <div class="result-value">
                {prediction:.4f}
                </div>

                <div class="result-unit">
                kcal mol⁻¹
                </div>

                <br>

                <b>Predicted Solvation Free Energy (ΔG)</b>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # Input summary
            # ------------------------------------------------

            st.divider()

            st.subheader("Prediction Summary")

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:

                st.write("**Solute**")
                st.code(solute)

            with summary_col2:

                st.write("**Solvent**")
                st.code(solvent)


        except Exception:

            st.error(
                "❌ Prediction failed. "
                "Please check the SMILES strings and model files."
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧪 Hybrid GNN")

    st.write(
        "Single-pair solvation free energy prediction."
    )

    st.divider()

    st.subheader("Model")

    st.write(
        f"**Input features:** {input_features}"
    )

    st.write(
        f"**Hidden dimension:** {config['hidden_dim']}"
    )

    st.write(
        f"**Attention heads:** {config['heads']}"
    )

    st.divider()

    st.subheader("Input")

    st.write(
        "• Solute SMILES"
    )

    st.write(
        "• Solvent SMILES"
    )

    st.divider()

    st.caption(
        "Hybrid GNN-based solvation free energy prediction"
    )
