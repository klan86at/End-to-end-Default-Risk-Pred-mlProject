from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
import logging


# importing configuration and utils
from defaultMlProj.config.configuration import ConfigurationManager
from defaultMlProj.utils.common import load_model

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Pydantic Models 

class CustomerData(BaseModel):
    credit_score:  float = Field(..., gt=300, lt=850, description="Credit score between 300-850")        
    income: float = Field(..., ge=0, description="Annual income in Kes")                 
    loan_amount: float = Field(..., gt=0, description="Loan term in moths")
    loan_term: float = Field(..., ge=12, le=360, description="Loan term in Months")
    interest_rate: float = Field(..., ge=0.1, le=36.0, description="Interest rate in %")
    debt_to_income_ratio: float = Field(..., ge=0, le=1, description="DTI ratio (0-1)")
    employment_years: float = Field(..., ge=0, le=50, description="Years of employment")
    savings_balance: float = Field(..., ge=0, description="Savings account balance")
    age: float = Field(..., ge=18, le=100, description="Customer age")


class PredictionResponse(BaseModel):
    predicted_default_risk_score: float
    risk_level: str
    threshold: float


# Global variable to hold the model
model = None
pred_threshold = 0.6
feat_order = [
    "credit_score", "income", "loan_amount", "loan_term",
    "interest_rate", "debt_to_income_ratio", "employment_years",
    "savings_balance", "age"
]

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle startup and shutdown events"""
    global model

    try:
        config = ConfigurationManager()
        model_serving_config = config.get_model_serving_config()
        logger.info(f"Raw model_path from config: {model_serving_config.model_path}")
        logger.info(f"Type: {type(model_serving_config.model_path)}")
        # Load Model
        model_path = Path(model_serving_config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

    except Exception as e:
        logger.error(f"Failed to load model at startup: {e}")
        raise

    yield 

    # Shutdown: cleanup
    print("Shutting down API...")

# FastAPI App

app = FastAPI(
    title="Default Risk Prediction API",
    description="Predict loan default risk score using a trianed model built with FastAPI & scikit-learn",
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name":"klan86at",
        "url": "https://github.com/klan86at",
    },
    license_info={
        "name":"MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)
# ================
# 3. API Endpoints
# ================

@app.get("/", tags=["Home"])
def home():
    return{
        "message": "Welcome to the Default Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict"
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(customer: CustomerData):
    """
    Predict default risk for a single customer.
    The input is validated, ordered, and passed through the full stacking pipeline.
    """
    try:
        input_dict = customer.model_dump()
        input_df = pd.DataFrame([customer.model_dump()])

        # Enforcing col order to match training
        try:
            input_df = input_df[feat_order]
        except KeyError as e:
            logger.error(f"Missing or extra columns in input data: {e}")
            raise HTTPException(status_code=400, detail=f"Missing required fields: {e}")
        
        # Predict
        try:
            prediction = model.predict(input_df)[0]
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        # Determining risk level
        risk_level = (
            "High" if prediction >= pred_threshold else
            "Medium" if prediction >= 0.2 else
            "Low"
        )
        result = {
            "predicted_default_risk_score": round(float(prediction), 4),
            "risk_level": risk_level,
            "threshold": pred_threshold
        }
        logger.info(f"Prediction result: {result}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
# Version endpoint
@app.get("/version", tags=["Meta"])
def version():
    return {"version": "1.0.0", "status": "single-prediction-only"}