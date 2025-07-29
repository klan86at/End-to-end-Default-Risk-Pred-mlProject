# app.py
import streamlit as st
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import traceback

# --- Page Config ---
st.set_page_config(page_title="Loan Risk Predictor", layout="centered")

# --- Title ---
st.title("🏦 Loan Default Risk Prediction")
st.markdown("Upload customer data or enter details to predict default risk.")

# --- Helper: Retry Session ---
def get_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# --- 1. Single Prediction ---
st.header("👤 Single Customer Prediction")

with st.form("single_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 700)
        income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=10000)
        loan_term = st.slider("Loan Term (months)", 12, 360, 60)
        interest_rate = st.number_input("Interest Rate (%)", 0.1, 36.0, 5.5)
    with col2:
        dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
        employment_years = st.slider("Employment Years", 0, 50, 5)
        savings = st.number_input("Savings Balance ($)", min_value=0, value=10000)
        age = st.slider("Age", 18, 100, 35)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    api_url = "https://default-risk-api.onrender.com/predict"  # ✅ No trailing space

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
        session = get_session()
        response = session.post(api_url, json=data, timeout=60)

        if response.status_code == 200:
            result = response.json()
            risk_score = result["predicted_default_risk_score"]
            risk_level = result["risk_level"]

            st.success(f"✅ Predicted Risk Score: **{risk_score:.4f}**")
            st.markdown(f"### Risk Level: `{risk_level}`")

            # Download single result
            df = pd.DataFrame([data])
            df["predicted_default_risk_score"] = risk_score
            df["risk_level"] = risk_level
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Prediction", csv, "prediction.csv", "text/csv")
        else:
            st.error(f"❌ API Error: {response.status_code}")
            st.code(response.text)

    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. The API may be waking up. Try again in 1 minute.")
    except requests.exceptions.ConnectionError:
        st.error("🔗 Connection failed. Is the backend live?")
    except Exception as e:
        st.error(f"🚨 Unexpected error: {e}")


# --- 2. Batch Prediction (Upload CSV) ---
st.header("📊 Batch Prediction (Upload CSV)")

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
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())

        required_columns = [
            "credit_score", "income", "loan_amount", "loan_term",
            "interest_rate", "debt_to_income_ratio", "employment_years",
            "savings_balance", "age"
        ]

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        # Convert to float and handle missing values
        df[required_columns] = df[required_columns].astype(float).fillna(0)

        if st.button("🚀 Predict for All Customers"):
            with st.spinner("Processing predictions..."):
                predictions = []
                api_url = "https://default-risk-api.onrender.com/predict/batch"

                # Add retry logic
                session = requests.Session()
                retry = Retry(
                    total=5,
                    backoff_factor=2,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                adapter = HTTPAdapter(max_retries=retry)
                session.mount("http://", adapter)
                session.mount("https://", adapter)

                # Prepare payload as a list of dictionaries
                payload = df[required_columns].to_dict(orient="records")

                try:
                    response = session.post(api_url, json=payload, timeout=60)
                    if response.status_code == 200:
                        result = response.json()
                        pred_df = pd.DataFrame(result["predictions"])
                        result_df = pd.concat([df, pred_df], axis=1)
                        st.write("### ✅ Batch Predictions")
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

                    else:
                        st.error(f"Batch API Error: {response.status_code}")
                        st.code(response.text)

                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error reading CSV: {str(e)}")
        st.code(traceback.format_exc())