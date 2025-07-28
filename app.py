# 1. Single customer prediction
# app.py

import streamlit as st
import requests
import pandas as pd

# Title
st.title("Loan Default Risk Prediction")

# Input form
st.header("Enter Customer Details")

credit_score = st.slider("Credit Score", 300, 850, 700)
income = st.number_input("Annual Income ($)", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=10000)
loan_term = st.slider("Loan Term (months)", 12, 360, 60)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.1, max_value=36.0, value=5.5)
dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
employment_years = st.slider("Employment Years", 0, 50, 5)
savings = st.number_input("Savings Balance ($)", min_value=0, value=10000)
age = st.slider("Age", 18, 100, 35)

# Predict button
if st.button("Predict Risk"):
    # Call your FastAPI
    api_url = "https://default-risk-api.onrender.com/predict"  # ← Use your Render URL

    data = {
        "credit_score": credit_score,
        "income": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "interest_rate": interest_rate,
        "debt_to_income_ratio": dti,
        "employment_years": employment_years,
        "savings_balance": savings,
        "age": age
    }

    try:
        response = requests.post(api_url, json=data)
        if response.status_code == 200:
            result = response.json()
            risk_score = result["predicted_default_risk_score"]
            risk_level = result["risk_level"]

            st.success(f"Predicted Risk Score: {risk_score:.4f}")
            st.markdown(f"### Risk Level: `{risk_level}`")
        else:
            st.error("Prediction failed")
    except Exception as e:
        st.error(f"Connection error: {e}")


# 2. --- Batch Prediction Section ---

with open("sample_input.csv", "w") as f:
    f.write("credit_score,income,loan_amount,loan_term,interest_rate,debt_to_income_ratio,employment_years,savings_balance,age\n")
    f.write("750,50000,10000,36,5.5,0.3,5,10000,35\n")
    f.write("700,60000,15000,60,6.0,0.4,3,5000,40")

with open("sample_input.csv", "r") as f:
    st.download_button("🔽 Download Sample CSV Template", f, "sample_input.csv")

st.header("Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload a CSV file with customer data", type="csv")

if uploaded_file is not None:
    # Read data
    batch_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(batch_df.head())

    if st.button("Predict for All Customers"):
        # Validate required columns
        required_columns = [
            "credit_score", "income", "loan_amount", "loan_term",
            "interest_rate", "debt_to_income_ratio", "employment_years",
            "savings_balance", "age"
        ]

        if not all(col in batch_df.columns for col in required_columns):
            st.error(f"CSV must contain all required columns: {required_columns}")
        else:
            with st.spinner("Predicting..."):
                predictions = []
                api_url = "https://default-risk-api.onrender.com/predict/batch"

                for _, row in batch_df.iterrows():
                    data = row[required_columns].to_dict()
                    try:
                        response = requests.post(api_url, json=data)
                        if response.status_code == 200:
                            pred = response.json()
                            predictions.append({
                                "predicted_default_risk_score": pred["predicted_default_risk_score"],
                                "risk_level": pred["risk_level"]
                            })
                        else:
                            predictions.append({
                                "predicted_default_risk_score": None,
                                "risk_level": "Error"
                            })
                    except:
                        predictions.append({
                            "predicted_default_risk_score": None,
                            "risk_level": "Connection Error"
                        })

                # Add predictions to DataFrame
                result_df = batch_df.copy()
                pred_df = pd.DataFrame(predictions)
                result_df = pd.concat([result_df, pred_df], axis=1)

                st.write("### Batch Predictions")
                st.dataframe(result_df)

                # Add download button
                @st.cache_data
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df(result_df)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predicted_risk_scores.csv",
                    mime="text/csv"
                )