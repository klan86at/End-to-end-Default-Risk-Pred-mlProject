# app.py
import streamlit as st
import requests
import pandas as pd
import time

# Title
st.title("Loan Default Risk Prediction")

# --- 1. Single Customer Prediction ---
st.header("👤 Single Customer Prediction")

# Input form
credit_score = st.slider("Credit Score", 300, 850, 700)
income = st.number_input("Annual Income ($)", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=10000)
loan_term = st.slider("Loan Term (months)", 12, 360, 60)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.1, max_value=36.0, value=5.5)
dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
employment_years = st.slider("Employment Years", 0, 50, 5)
savings = st.number_input("Savings Balance ($)", min_value=0, value=10000)
age = st.slider("Age", 18, 100, 35)

# Buttons
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("Predict Risk")
with col2:
    if st.button("Clear Inputs"):
        st.rerun()

if predict_btn:
    api_url = "https://default-risk-api.onrender.com/predict"

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
        # Add timeout to avoid hanging
        response = requests.post(api_url, json=data, timeout=30)
        
        if response.status_code == 200:
            try:
                result = response.json()
                risk_score = result["predicted_default_risk_score"]
                risk_level = result["risk_level"]

                st.success(f"Predicted Risk Score: {risk_score:.4f}")
                st.markdown(f"### Risk Level: `{risk_level}`")

                # Download single result
                single_df = pd.DataFrame([data])
                single_df["predicted_default_risk_score"] = risk_score
                single_df["risk_level"] = risk_level
                csv = single_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download This Prediction",
                    data=csv,
                    file_name="single_prediction.csv",
                    mime="text/csv"
                )
            except ValueError:
                st.error("Invalid JSON response from API")
        else:
            st.error(f"API Error: {response.status_code}")
            st.code(response.text)
    except requests.exceptions.Timeout:
        st.error("Request timed out. The API might be slow to wake up.")
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to API. Check if the backend is live.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


# --- 2. Batch Prediction ---

st.header("Batch Prediction (Upload CSV)")

# Sample CSV template
sample_data = """credit_score,income,loan_amount,loan_term,interest_rate,debt_to_income_ratio,employment_years,savings_balance,age
750,50000,10000,36,5.5,0.3,5,10000,35
700,60000,15000,60,6.0,0.4,3,5000,40"""

st.download_button(
    "🔽 Download Sample CSV Template",
    data=sample_data,
    file_name="sample_input.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader("Upload a CSV file with customer data", type="csv")

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(batch_df.head())

        # Clear upload button
        if st.button("🗑️ Clear Upload"):
            st.rerun()

        required_columns = [
            "credit_score", "income", "loan_amount", "loan_term",
            "interest_rate", "debt_to_income_ratio", "employment_years",
            "savings_balance", "age"
        ]

        missing_cols = set(required_columns) - set(batch_df.columns)
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            if st.button("Predict for All Customers"):
                with st.spinner("Processing batch predictions..."):
                    predictions = []
                    api_url = "https://default-risk-api.onrender.com/predict/batch"  # ✅ No trailing space

                    for idx, row in batch_df.iterrows():
                        data = row[required_columns].to_dict()
                        try:
                            response = requests.post(api_url, json=data, timeout=30)
                            if response.status_code == 200:
                                pred = response.json()
                                predictions.append({
                                    "predicted_default_risk_score": pred["predicted_default_risk_score"],
                                    "risk_level": pred["risk_level"]
                                })
                            else:
                                predictions.append({
                                    "predicted_default_risk_score": None,
                                    "risk_level": f"API Error {response.status_code}"
                                })
                        except requests.exceptions.Timeout:
                            predictions.append({
                                "predicted_default_risk_score": None,
                                "risk_level": "Timeout"
                            })
                        except Exception as e:
                            predictions.append({
                                "predicted_default_risk_score": None,
                                "risk_level": f"Error: {str(e)}"
                            })

                    # Add predictions
                    result_df = batch_df.copy()
                    pred_df = pd.DataFrame(predictions)
                    result_df = pd.concat([result_df, pred_df], axis=1)

                    st.write("### Batch Predictions")
                    st.dataframe(result_df)

                    # Download options
                    @st.cache_data
                    def convert_df(df):
                        return df.to_csv(index=False).encode("utf-8")

                    csv = convert_df(result_df)
                    json_data = result_df.to_json(orient="records")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                    with col2:
                        st.download_button(
                            "📤 Download as JSON",
                            data=json_data,
                            file_name="predictions.json",
                            mime="application/json"
                        )

                    # Summary chart
                    if "risk_level" in result_df.columns:
                        st.write("### Prediction Summary")
                        risk_counts = result_df["risk_level"].value_counts()
                        st.bar_chart(risk_counts)

    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")