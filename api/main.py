from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, AsyncIterator
from contextlib import asynccontextmanager
import pandas as pd
import logging

# importing config and utils
from defaultMlProj.config.configuration import ConfigurationManager
from defaultMlProj.utils.common import load_model

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Pydantic Models (Input/Output)

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

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]



# Global variable to hold the model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle startup and shutdown events"""
    global model
    try:
        config = ConfigurationManager()
        model_serving_config = config.get_model_serving_config()
        model = load_model(model_serving_config.model_path)
        logger.info("Model loaded succesfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield 

    # Shutdown: cleanup
    print("Shutting down...")

# 2.Load Model at start up

app = FastAPI(
    title="Default Risk Prediction API",
    description="Predict loan default risk score using model built with FastAPI & scikit-learn",
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

# 3. API Endpoints

@app.get("/", tags=["Home"])
def home():
    return{
        "message": "Welcome to the Default Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "batch": "POST /predict/batch"
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["Single Prediction"])
def predict_single(customer: CustomerData):
    """Predict default risk for a single customer"""
    try:
        input_df = pd.DataFrame([customer.model_dump()])

        # Predict
        prediction = model.predict(input_df)[0]
        

        logger.info(f"Prediction: {prediction:.4f}")

        return {
            "predicted_default_risk_score": round(float(prediction), 4)
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
@app.post("/predict/batch")
def predict_batch(customers: List[CustomerData]):
    try:
        input_df = pd.DataFrame([c.model_dump() for c in customers])
        predictions = model.predict(input_df)
        
        return {
            "predictions": [
                {"predicted_default_risk_score": round(float(pred), 4)}
                for pred in predictions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")