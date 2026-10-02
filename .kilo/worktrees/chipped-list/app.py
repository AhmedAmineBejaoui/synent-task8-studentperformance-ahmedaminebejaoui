"""Student Performance Prediction — Streamlit demo.

User input -> predicted final grade G3 (Random Forest, clipped to 0-20).
Model file: student_model_v1.pkl (built by Task8_StudentPerformance_ML.ipynb).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILE = Path(__file__).with_name("student_model_v1.pkl")

TEST_RMSE = 1.204
TEST_R2 = 0.851

# Most common profile in the training data; advanced fields keep these defaults.
TYPICAL = {
    'school': 'GP', 'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T',
    'Mjob': 'other', 'Fjob': 'other', 'reason': 'course', 'guardian': 'mother',
    'traveltime': 1, 'famsup': 'yes', 'activities': 'no', 'nursery': 'yes',
    'higher': 'yes', 'internet': 'yes', 'famrel': 4, 'freetime': 3,
    'goout': 3, 'Dalc': 1, 'Walc': 1, 'health': 5,
}


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_FILE)


st.set_page_config(page_title="Student Grade Prediction", layout="centered")
st.title("Student Final Grade (G3) Prediction")
st.write(
    "Predict a student's final grade (0–20) from background and prior grades. "
    f"Model: Random Forest (test RMSE {TEST_RMSE}, R² {TEST_R2})."
)

artifact = load_artifact()
model = artifact["model"]
features = artifact["features"]
q_low, q_high = artifact["residual_quantiles_10_90"]

col1, col2 = st.columns(2)
with col1:
    g1 = st.slider("First-period grade (G1)", 0, 20, 11)
    g2 = st.slider("Second-period grade (G2)", 0, 20, 11)
    studytime = st.selectbox("Weekly study time", [1, 2, 3, 4], index=1,
                             format_func=lambda v: {1: "< 2h", 2: "2–5h", 3: "5–10h", 4: "> 10h"}[v])
    failures = st.selectbox("Past class failures", [0, 1, 2, 3], index=0)
    absences = st.number_input("School absences", value=2, min_value=0, max_value=60)
    age = st.number_input("Age", value=17, min_value=15, max_value=22)
with col2:
    sex = st.selectbox("Sex", ["F", "M"], index=0)
    medu = st.selectbox("Mother's education (0–4)", [0, 1, 2, 3, 4], index=2)
    fedu = st.selectbox("Father's education (0–4)", [0, 1, 2, 3, 4], index=2)
    schoolsup = st.selectbox("Extra school support", ["yes", "no"], index=1)
    paid = st.selectbox("Extra paid classes", ["yes", "no"], index=1)
    romantic = st.selectbox("In a relationship", ["yes", "no"], index=1)

if st.button("Predict"):
    row = dict(TYPICAL)
    row.update({"sex": sex, "age": age, "Medu": medu, "Fedu": fedu,
                "studytime": studytime, "failures": failures, "absences": absences,
                "schoolsup": schoolsup, "paid": paid, "romantic": romantic,
                "G1": g1, "G2": g2})
    new_df = pd.DataFrame([row])
    pred = float(np.clip(model.predict(new_df[features]), 0, 20)[0])
    low = float(np.clip(pred + q_low, 0, 20))
    high = float(np.clip(pred + q_high, 0, 20))

    st.metric("Predicted final grade (G3)", round(pred, 1))
    st.write(f"80% interval: **{low:.1f} – {high:.1f}**.")
    if pred < 10:
        st.warning("Below the passing threshold (10): at-risk student.")
