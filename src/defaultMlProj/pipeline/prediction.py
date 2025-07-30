# prediction.py
import pandas as pd
import joblib
from pathlib import Path
from defaultMlProj.constants.constant import *
from defaultMlProj.utils.common import read_yaml, load_model
from defaultMlProj.config.configuration import ConfigurationManager

# Constants
pred_threshold = 0.6
required_features = [
    "credit_score", "income", "loan_amount", "loan_term",
    "interest_rate", "debt_to_income_ratio", "employment_years",
    "savings_balance", "age"
]

def predict_single(customer_data: dict, model, preprocessor) -> dict:
    """ Predicts the risk for a single customer.
    Args:
        customer_data (dict): Customer data with features.
        model: Trained model object.
        preprocessor: Fitted ColumnTransfer for scaling.
    Returns:
        dict: Prediction result with risk score and label.
    """
    # Validate input
    missing = [k for k in required_features if k not in customer_data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    # Create DataFrame
    input_df = pd.DataFrame([customer_data])
    
    # Confirm column oreder matches training
    try:
        input_df = input_df[required_features]
    except KeyError as e:
        raise KeyError(f"Invalid input schema: {e}")
    
    # Apply preprocessing (scaling)
    try:
        transformed_input = preprocessor.transform(input_df)
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {e}")
    
    # Predict
    try:
        prediction = model.predict(transformed_input)[0]
        probability = (
            model.predict_proba(transformed_input)[0].tolist()
            if hasattr(model, "predict_proba")
            else None
        )
    except Exception as e:
        raise RuntimeError(f"Model Prediction failed: {e}")
    
    # Generate result
    risk_level = "High" if prediction >= pred_threshold else "Low"

    results = {
        "predicted_default_risk_score": round(float(prediction), 4),
        "risk_level": risk_level,
        "threshold_used": pred_threshold,
        "probability": probability  # Optional: class probabilities
    }
    
    return results


if __name__ == "__main__":
    # Load cnfiguration and artifacts
    config = ConfigurationManager()
    serving_config = config.get_model_serving_config()

    # Load model and preprocessor
    model = load_model(serving_config.model_path)
    preprocessor_path = Path(serving_config.model_dir) / "preprocessor.pkl"
    preprocessor = joblib.load(preprocessor_path)

    print(f"Loaded model from {serving_config.model_path}")
    print(f"Loaded preprocessor from {preprocessor_path}")

    # Testing single prediction
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

    try:
        result = predict_single(sample_data, model, preprocessor)
        print(f"\n Single Customer Prediction: {result}")
    except Exception as e:
        print(f"\n Prediction failed: {e}")