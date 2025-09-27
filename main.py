import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Load model and encoders
# -----------------------------
loaded_model = joblib.load("model.pkl")

# Extract model if stored inside a dictionary
if isinstance(loaded_model, dict) and "model" in loaded_model:
    model = loaded_model["model"]
else:
    model = loaded_model

# Load encoders
try:
    encoders = joblib.load("encoders.pkl")
except Exception:
    encoders = None
    st.info("No encoders.pkl found — categorical encoding will use fallback mapping.")

# -----------------------------
# Feature setup
# -----------------------------
categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Correct training columns
training_columns = [
    'gender','SeniorCitizen','Partner','Dependents','tenure',
    'PhoneService','MultipleLines','InternetService','OnlineSecurity',
    'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
    'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
    'MonthlyCharges','TotalCharges'
]

# -----------------------------
# Safe transform function
# -----------------------------
def safe_transform(le, series):
    known_classes = set(le.classes_)
    return series.map(lambda x: le.transform([x])[0] if x in known_classes else -1)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction")
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details below and click Predict.")

# --- Inputs ---
input_data = {}
input_data["gender"] = st.selectbox("Gender", ["Male", "Female"])
input_data["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1])
input_data["Partner"] = st.selectbox("Partner", ["Yes", "No"])
input_data["Dependents"] = st.selectbox("Dependents", ["Yes", "No"])
input_data["tenure"] = st.number_input("Tenure (months)", min_value=0, max_value=1000, value=12)
input_data["PhoneService"] = st.selectbox("PhoneService", ["Yes", "No"])
input_data["MultipleLines"] = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])
input_data["InternetService"] = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
input_data["OnlineSecurity"] = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
input_data["OnlineBackup"] = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
input_data["DeviceProtection"] = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
input_data["TechSupport"] = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
input_data["StreamingTV"] = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
input_data["StreamingMovies"] = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
input_data["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
input_data["PaperlessBilling"] = st.selectbox("PaperlessBilling", ["Yes", "No"])
input_data["PaymentMethod"] = st.selectbox("PaymentMethod", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
input_data["MonthlyCharges"] = st.number_input("MonthlyCharges", min_value=0.0, max_value=10000.0, value=70.0)
input_data["TotalCharges"] = st.number_input("TotalCharges", min_value=0.0, max_value=100000.0, value=500.0)

st.divider()

# --- Predict button ---
if st.button("Predict"):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    if encoders:
        for col in categorical_cols:
            if col in input_df.columns and col in encoders:
                input_df[col] = safe_transform(encoders[col], input_df[col])
    else:
        # Fallback: simple mapping for Yes/No and categorical
        yes_no_map = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        internet_map = {"DSL": 1, "Fiber optic": 2, "No": 0}
        payment_map = {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3
        }
        input_df["gender"] = input_df["gender"].map({"Male": 0, "Female": 1})
        input_df["Partner"] = input_df["Partner"].map(yes_no_map)
        input_df["Dependents"] = input_df["Dependents"].map(yes_no_map)
        input_df["PhoneService"] = input_df["PhoneService"].map(yes_no_map)
        input_df["MultipleLines"] = input_df["MultipleLines"].map(yes_no_map)
        input_df["InternetService"] = input_df["InternetService"].map(internet_map)
        input_df["OnlineSecurity"] = input_df["OnlineSecurity"].map(yes_no_map)
        input_df["OnlineBackup"] = input_df["OnlineBackup"].map(yes_no_map)
        input_df["DeviceProtection"] = input_df["DeviceProtection"].map(yes_no_map)
        input_df["TechSupport"] = input_df["TechSupport"].map(yes_no_map)
        input_df["StreamingTV"] = input_df["StreamingTV"].map(yes_no_map)
        input_df["StreamingMovies"] = input_df["StreamingMovies"].map(yes_no_map)
        input_df["Contract"] = input_df["Contract"].map(contract_map)
        input_df["PaperlessBilling"] = input_df["PaperlessBilling"].map(yes_no_map)
        input_df["PaymentMethod"] = input_df["PaymentMethod"].map(payment_map)

    # Ensure column order matches training
    input_df = input_df[training_columns]

    # Prediction
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else "N/A"
        result = "Yes (Churn)" if prediction == 1 else "No (Not Churn)"
        st.success(f"Prediction: {result}")
        st.info(f"Churn Probability: {prob}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
