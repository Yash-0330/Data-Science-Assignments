"""
Streamlit app for diabetes risk prediction.
Loads the full sklearn Pipeline (imputer + scaler + logistic regression)
saved in §3 and applies it to user inputs at inference time.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# 1) Load artifacts once per container (critical for speed)
#    @st.cache_resource prevents re-loading on every widget interaction
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "diabetes_model.pkl")
    meta_path = BASE_DIR / "diabetes_model_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, meta

model, meta = load_artifacts()

# Features the model was trained on (enforced column order)
FEATURES = meta.get(
    "feature_names",
    ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
)

# ----------------------------------------------------------------------
# 2) Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.write(
    "Enter the patient's clinical measurements below. "
    "The model estimates the **probability of diabetes** (Outcome = 1). "
    "This is a decision-support tool, not a diagnosis."
)

# ----------------------------------------------------------------------
# 3) Input form
#    Use NaN for values the user leaves blank (0 is a valid value for
#    Pregnancies but not for the other clinical measurements).
# ----------------------------------------------------------------------
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70, step=1)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)

    with col2:
        insulin = st.number_input("Insulin (µU/mL)", min_value=0, max_value=900, value=80, step=1)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.2f")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)

    submitted = st.form_submit_button("Predict risk", use_container_width=True)

# ----------------------------------------------------------------------
# 4) Prediction
# ----------------------------------------------------------------------
if submitted:
    # Build a single-row DataFrame in the exact training column order.
    # Blank/zero values become NaN so the pipeline's median imputer
    # (trained in §2) handles them the same way it did during training.
    user_row = {
        "Pregnancies": pregnancies,
        "Glucose": glucose if glucose > 0 else np.nan,
        "BloodPressure": blood_pressure if blood_pressure > 0 else np.nan,
        "SkinThickness": skin_thickness if skin_thickness > 0 else np.nan,
        "Insulin": insulin if insulin > 0 else np.nan,
        "BMI": bmi if bmi > 0 else np.nan,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    X_input = pd.DataFrame([user_row])[FEATURES]   # enforce order

    try:
        proba = float(model.predict_proba(X_input)[0, 1])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    # Optional: allow a custom threshold for screening vs. confirmation.
    threshold = st.session_state.get("threshold", 0.5)
    predicted_class = int(proba >= threshold)

    # ------------------------------------------------------------------
    # 5) Display result
    # ------------------------------------------------------------------
    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Predicted probability", f"{proba:.1%}")
    c2.metric("Classification", "Diabetes" if predicted_class else "No diabetes")
    st.progress(min(max(proba, 0.0), 1.0))

    if proba >= 0.5:
        st.warning("High risk — recommend clinical follow-up and confirmatory testing.")
    else:
        st.success("Low risk based on this model. Routine monitoring advised.")

    st.caption(
        f"Decision threshold: {threshold:.2f}. "
        "Lowering the threshold increases sensitivity (catches more diabetics) "
        "at the cost of more false alarms — appropriate for screening."
    )

    with st.expander("View input summary"):
        st.dataframe(X_input.T.rename(columns={0: "Value"}))