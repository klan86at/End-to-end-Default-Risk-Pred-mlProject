import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from defaultMlProj.constants.constant import *
from defaultMlProj.utils.common import read_yaml, load_model
from defaultMlProj.config.configuration import ConfigurationManager



def predict_single(customer_data: dict) -> dict:
    """ Predicts the risk for a single customer.
    Args:
        customer_data (dict): Customer data with features.
    Returns:
        dict: Prediction result with risk score and label.
    """
    # Load config to get model_path
    config = ConfigurationManager()
    serving_config = config.get_model_serving_config()
    model_path = serving_config.model_path
    
    # Load the model
    model = load_model(model_path)

    # Predict
    input_df = pd.DataFrame([customer_data])
    prediction = model.predict(input_df)[0]

    return {
    "predicted_default_risk_score": float(prediction),
    "risk_level": "High" if prediction > 0.7 else "Low"
    }

def predict_batch(input_data: pd.DataFrame) -> pd.DataFrame:
    """ Predicts the risk for a batch of customers.
    Args:
        input_data (pd.DataFrame): DataFrame with customer features.
    Returns:
        pd.DataFrame: DataFrame with predictions and risk scores.
    """
    # To check for correct format
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    # Load config to get model_path
    config = ConfigurationManager()
    serving_config = config.get_model_serving_config()
    model_path = serving_config.model_path

    # Load the model
    model = load_model(model_path)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Add predictions to the DataFrame
    input_data['predicted_default_risk_score'] = predictions
    
    return input_data[['predicted_default_risk_score']]

if __name__ == "__main__":
    
    sample_data = {
       "credit_score": 750,
        "income": 50000,
        "loan_amount": 10000,
        "loan_term": 36,
        "interest_rate": 5.5,
        "debt_to_income_ratio": 0.3,
        "employment_years": 5,
        "savings_balance": 10000,
        "age": 35
    }
    
    # Predict for a single customer
    result = predict_single(sample_data)
    print(f"Single Customer Prediction: {result:.4f}")
    
    # Predict for a batch of customers
    batch_data = pd.DataFrame([sample_data, sample_data])  # Example batch data
    batch_predictions = predict_batch(batch_data)
    print(f"Batch Predictions:\n{batch_predictions}")