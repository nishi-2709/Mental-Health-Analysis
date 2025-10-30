# app.py - Streamlit app to load pipeline and get predictions
import streamlit as st
import pickle
import pandas as pd

MODEL_PATH = "best_model.pkl" 

@st.cache_resource
def load_model(path=best_model.pkl):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.set_page_config(page_title="Mental Health Prediction", layout="centered")
st.title("Mental Health in Tech — Prediction demo")

st.markdown("Fill the form and click **Predict**. This uses the saved preprocessing + model pipeline.")

# Build inputs automatically from training data columns if available in pipeline
# Best practice: list the expected input columns; user should adapt these to match training features.
# Here we try to infer input names from pipeline if it exists:
try:
    # If pipeline was saved as sklearn Pipeline with ColumnTransformer first
    preprocessor = model.named_steps.get('preprocessor', None)
    # If we cannot infer, fall back to manual list for common fields
except Exception:
    preprocessor = None

# We'll ask a few common fields from the original dataset. Adjust fields to match your dataset.
age = st.number_input("Age", min_value=16, max_value=120, value=30)
gender = st.selectbox("Gender", options=["Male", "Female", "Other", "Prefer not to say"])
country = st.text_input("Country", value="United States")
self_employed = st.selectbox("Self-employed?", options=["yes", "no", "nan"])
family_history = st.selectbox("Family history of mental illness?", options=["yes","no"])
work_interfere = st.selectbox("Does work interfere with mental health?", options=["Never","Rarely","Sometimes","Often"])
remote_work = st.selectbox("Do you work remotely?", options=["yes","no"])
tech_company = st.text_input("Company name (optional)")

if st.button("Predict"):
    # Create a DataFrame with the same column names used in training.
    # IMPORTANT: adjust column names to exactly match the training DataFrame's columns.
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Country": country,
        "self_employed": self_employed,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "remote_work": remote_work,
        "Company": tech_company
    }])
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") or (hasattr(model.named_steps['clf'], 'predict_proba')) else None
        st.write("### Prediction")
        st.write("Likely sought mental health treatment:" , "Yes 🔴" if pred==1 else "No ✅")
        if proba is not None:
            st.write(f"Confidence (probability): {proba:.2f}")
    except Exception as e:
        st.error("Model prediction failed. You need to ensure the input columns match the exact columns used for training.")
        st.write("Error:", e)

st.markdown("---")
st.markdown("""
---
**Developed by:** Nishi Singh 
**Project:** Machine Learning Mini Project — Mental Health Analysis
""")