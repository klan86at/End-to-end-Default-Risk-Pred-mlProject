# app.py
import streamlit as st
import requests
import pandas as pd
import traceback
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Page Configuration
st.set_page_config(page_title="Loan Risk Predictor", layout="centered")
# =============
# --- Title ---
# =============

st.title("🏦 Loan Default Risk Prediction")
st.markdown("Upload customer data or enter details to predict default risk.")

# Constants
API_BASE_URL = "https://default-risk-api.onrender.com"
SINGLE_PREDICT_URL = f"{API_BASE_URL}/predict"
TIMEOUT = 60

# Helper: Retry Session 
def get_session():
    """Create a requests session with retry logic for reliability"""
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

# 1. Single Prediction 
st.header("👤 Single Customer Prediction")

with st.form("Prediction_form"):
    st.markdown("### 🔹 Customer Information")

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

    submitted = st.form_submit_button("Predict Risk", type="secondary")

if submitted:
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
        st.info("📡 Sending request to API...")
        session = get_session()
        response = session.post(SINGLE_PREDICT_URL, json=data, timeout=TIMEOUT)

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

            st.download_button(
                "📥 Download Prediction", 
                data=csv,
                file_name="prediction.csv",
                mime="text/csv"
            )
        elif response.status_code == 422:
            st.error("⚠️ Invalid input: Please check all values are within range.")
            try:
                details = response.json()
                st.code(details, language="json")
            except Exception:
                st.text("No details available.")
        else:
            st.error(f"API Error {response.status_code}")
            st.code(response.text)

    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. API may be waking up. Try again in 1 minute.")
    except requests.exceptions.ConnectionError:
        st.error("🔗 Connection failed. Is the backend live?")
    except Exception as e:
        st.error("Unexpected error occurred")
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("Powered by FastAPI & Streamlit | model: Stacking Reggressor")