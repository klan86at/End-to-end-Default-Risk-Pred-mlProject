from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Config for data ingestion.
    Since we're using a local file.
    """
    root_dir: Path
    source_data_file: Path  # Local path to your CSV (notebook/data/default.csv)
    local_data_file: Path   # Where to copy it (artifacts/data_ingestion/default.csv)


@dataclass(frozen=True)
class DataValidationConfig:
    """
    Config for data validation.
    Checks if the data conforms to expected schema, types, etc.
    """
    root_dir: Path
    data_path: Path           # Path to the ingested data (artifacts/data_ingestion/default.csv)
    status_file: Path         # Output file to log validation status
    all_schema: Dict          # Expected column names and types (from params/schema.yaml)
    validation_status: str = "validation_status"  # Optional key name


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Config for transforming data (e.g., train/test split, scaling, encoding).
    """
    root_dir: Path
    data_path: Path           # Input: validated data
    # Optional: Add transformer_name, scaling_method,  if needed later


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Config for training the model.
    """
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str           # "model.joblib"
    target_column: str        # Name of the target variable



@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Config for evaluating the model.
    """
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path    # artifacts/model_evaluation/metrics.json
    all_params: Dict          # Model parameters (from params.yaml)
    target_column: str
    mlflow_uri: str           # e.g "https://mlflow.example.com " or "mlruns"