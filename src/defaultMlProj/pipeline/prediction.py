# prediction.py
import pandas as pd
from pathlib import Path
from defaultMlProj.constants.constant import *
from defaultMlProj.utils.common import read_yaml, load_model
from defaultMlProj.config.configuration import ConfigurationManager


def predict_single(customer_data: dict, model) -> dict:
    """ Predicts the risk for a single customer.
    Args:
        customer_data (dict): Customer data with features.
        model: Trained model object.
    Returns:
        dict: Prediction result with risk score and label.
    """
    required_keys = [
        "credit_score", "income", "loan_amount", "loan_term",
        "interest_rate", "debt_to_income_ratio", "employment_years",
        "savings_balance", "age"
    ]
    if not all(k in customer_data for k in required_keys):
        raise ValueError("Missing required fields")

    input_df = pd.DataFrame([customer_data])
    prediction = model.predict(input_df)[0]
    
    return {
        "predicted_default_risk_score": round(float(prediction), 4),
        "risk_level": "High" if prediction > 0.7 else "Low"
    }


def predict_batch(input_data: pd.DataFrame, model) -> pd.DataFrame:
    """ Predicts the risk for a batch of customers.
    Args:
        input_data (pd.DataFrame): DataFrame with customer features.
        model: Trained model object.
    Returns:
        pd.DataFrame: DataFrame with predictions and risk levels.
    """
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    predictions = model.predict(input_data)
    
    result_df = pd.DataFrame()
    result_df['predicted_default_risk_score'] = predictions
    result_df['risk_level'] = [
        "High" if pred > 0.7 else "Low" for pred in predictions
    ]
    
    return result_df


if __name__ == "__main__":
    # Load model once
    config = ConfigurationManager()
    serving_config = config.get_model_serving_config()
    model = load_model(serving_config.model_path)

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

    # Predict single
    result = predict_single(sample_data, model)
    print(f"Single Customer Prediction: {result}")

    # Predict batch
    batch_data = pd.DataFrame([sample_data, sample_data])
    batch_predictions = predict_batch(batch_data, model)
    print(f"Batch Predictions:\n{batch_predictions}")