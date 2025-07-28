import streamlit as st
import requests
import pandas as pd
from pydantic import BaseModel, Field, conint, confloat
from typing import Optional
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# --- FastAPI Model Definition (Mirror exactly what your backend expects) ---
class LoanInput(BaseModel):
    credit_score: int = Field(..., ge=300, le=850, example=700)
    income: int = Field(..., ge=0, example=50000)
    loan_amount: int = Field(..., ge=1000, example=10000)
    loan_term: int = Field(..., ge=12, le=360, example=60)
    interest_rate: float = Field(..., ge=0.1, le=36.0, example=5.5)
    debt_to_income_ratio: float = Field(..., ge=0.0, le=1.0, example=0.3, alias="debt-to-income-ratio")
    employment_years: int = Field(..., ge=0, le=50, example=5)
    savings_balance: int = Field(..., ge=0, example=10000)
    age: int = Field(..., ge=18, le=100, example=35)

# --- API Config ---
API_SINGLE = "https://default-risk-api.onrender.com/predict"
API_BATCH = "https://default-risk-api.onrender.com/predict/batch"

st.set_page_config(page_title="Loan Default Risk")
st.title("Loan Default Risk Prediction")

# --------------------------------------------------
# 1) Single prediction with enhanced validation
# --------------------------------------------------
with st.expander("👤 Single Customer", expanded=True):
    credit_score = st.slider("Credit Score", 300, 850, 700)
    income = st.number_input("Annual Income ($)", 0, None, 50000)
    loan_amount = st.number_input("Loan Amount ($)", 1000, None, 10000)
    loan_term = st.slider("Loan Term (months)", 12, 360, 60)
    interest_rate = st.number_input("Interest Rate (%)", 0.1, 36.0, 5.5)
    dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    employment_years = st.slider("Employment Years", 0, 50, 5)
    savings = st.number_input("Savings Balance ($)", 0, None, 10000)
    age = st.slider("Age", 18, 100, 35)

    if st.button("Predict Risk"):
        try:
            # Validate against Pydantic model before sending
            payload = LoanInput(
                credit_score=credit_score,
                income=income,
                loan_amount=loan_amount,
                loan_term=loan_term,
                interest_rate=interest_rate,
                debt_to_income_ratio=dti,  # Pydantic handles alias
                employment_years=employment_years,
                savings_balance=savings,
                age=age
            ).model_dump(by_alias=True)  # Converts to API-ready format
            
            st.json(payload)  # Debug output
            
            r = requests.post(API_SINGLE, json=payload, timeout=30)
            
            if r.status_code == 200:
                out = r.json()
                st.success(f"Score: {out['predicted_default_risk_score']:.4f}")
                st.info(f"Level: {out['risk_level']}")
            else:
                st.error(f"API error {r.status_code}")
                st.json(r.json())  # Show full error details
                
        except Exception as e:
            st.error(f"Validation error: {str(e)}")

# --------------------------------------------------
# 2) Batch prediction with robust validation
# --------------------------------------------------
with st.expander("Batch Prediction (CSV Upload)"):
    # Sample CSV with exactly matching column names
    sample = """credit_score,income,loan_amount,loan_term,interest_rate,debt_to_income_ratio,employment_years,savings_balance,age
750,50000,10000,36,5.5,0.3,5,10000,35
700,60000,15000,60,6.0,0.4,3,5000,40"""

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)
            st.dataframe(df.head())
            
            # 1. Column Alignment
            df = df.rename(columns={"debt_to_income_ratio": "debt-to-income-ratio"})
            
            # 2. Schema Validation
            required_cols = list(LoanInput.model_json_schema()["properties"].keys())
            if missing := set(required_cols) - set(df.columns):
                st.error(f"Missing columns: {missing}")
                st.stop()
                
            # 3. Type Conversion
            for col in required_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 4. Data Quality Check
            if df[required_cols].isna().any().any():
                st.error("Invalid values detected:")
                st.dataframe(df[df.isna().any(axis=1)])
                st.stop()
            
            if st.button("Predict all"):
                with st.spinner(f"Processing {len(df)} records..."):
                    # 5. Parallel Batch Processing
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        batches = [
                            batch.to_dict('records') 
                            for _, batch in df.groupby(np.arange(len(df)) // 50)
                        ]
                        
                        results = list(executor.map(
                            lambda b: requests.post(API_BATCH, json=b, timeout=60).json(),
                            batches
                        ))
                    
                    # 6. Combine Results
                    out = df.copy()
                    out[["risk_score", "risk_level"]] = pd.concat(
                        [pd.DataFrame(r) for r in results]
                    )
                    
                    st.success(f"Processed {len(out)} records")
                    st.dataframe(out)
                    
                    # 7. Enhanced Export
                    csv = out.to_csv(index=False)
                    st.download_button(
                        "📥 Download Full Results", 
                        csv,
                        "predictions.csv",
                        help="Contains original data + predictions"
                    )
        except Exception as e:
            st.error(f"Batch processing failed: {str(e)}")
            st.stop()

        except Exception as e:
            st.error(f"Batch processing failed: {str(e)}")
            st.stop()

# Add API documentation link
st.markdown("""
---
**API Documentation**:  
[View OpenAPI Spec](https://default-risk-api.onrender.com/docs) | 
[View Redoc](https://default-risk-api.onrender.com/redoc)
""")