# app.py
import streamlit as st
import requests
import pandas as pd

API_SINGLE = "https://default-risk-api.onrender.com/predict"
API_BATCH  = "https://default-risk-api.onrender.com/predict/batch"

st.set_page_config(page_title="Loan Default Risk")
st.title("Loan Default Risk Prediction")

# --------------------------------------------------
# 1) Single prediction
# --------------------------------------------------
with st.expander("👤 Single Customer"):
    credit_score        = st.slider("Credit Score", 300, 855, 700)
    income              = st.number_input("Annual Income ($)", 0, None, 50000)
    loan_amount         = st.number_input("Loan Amount ($)", 1000, None, 10000)
    loan_term           = st.slider("Loan Term (months)", 12, 360, 60)
    interest_rate       = st.number_input("Interest Rate (%)", 0.1, 36.0, 5.5)
    dti                 = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    employment_years    = st.slider("Employment Years", 0, 50, 5)
    savings             = st.number_input("Savings Balance ($)", 0, None, 10000)
    age                 = st.slider("Age", 18, 100, 35)

    if st.button("Predict Risk"):
        payload = dict(
            credit_score=credit_score,
            income=income,
            loan_amount=loan_amount,
            loan_term=loan_term,
            interest_rate=interest_rate,
            debt_to_income_ratio=dti,
            employment_years=employment_years,
            savings_balance=savings,
            age=age,
        )
        r = requests.post(API_SINGLE, json=payload, timeout=30)
        if r.status_code == 200:
            out = r.json()
            st.success(f"Score: {out['predicted_default_risk_score']:.4f}")
            st.info(f"Level: {out['risk_level']}")
        else:
            st.error(f"API error {r.status_code}: {r.text}")

# --------------------------------------------------
# 2) Batch prediction
# --------------------------------------------------
with st.expander("Batch Prediction (CSV Upload)"):

    sample = """credit_score,income,loan_amount,loan_term,interest_rate,debt_to_income_ratio,employment_years,savings_balance,age
750,50000,10000,36,5.5,0.3,5,10000,35
700,60000,15000,60,6.0,0.4,3,5000,40"""
    st.download_button("Download sample CSV", sample, "sample.csv")

file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())
    cols = ["credit_score","income","loan_amount","loan_term",
            "interest_rate","debt_to_income_ratio",
            "employment_years","savings_balance","age"]
    if not set(cols).issubset(df.columns):
        st.error("CSV missing required columns")
        st.stop()

    if st.button("🚀 Predict all"):
        payload = {"instances": df[cols].to_dict(orient="records")}
        r = requests.post("https://default-risk-api.onrender.com/predict/batch",
                          json=payload, timeout=60)
        if r.ok:
            preds = pd.DataFrame(r.json()["predictions"])
            out = pd.concat([df, preds], axis=1)
            st.dataframe(out)
            st.download_button("📥 Download CSV", out.to_csv(index=False), "batch.csv")
        else:
            st.error(f"{r.status_code} – {r.text}")