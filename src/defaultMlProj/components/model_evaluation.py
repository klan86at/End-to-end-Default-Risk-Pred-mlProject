# Model evaluation component
import os
import joblib
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from defaultMlProj import logger
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from defaultMlProj.entity.config_entity import ModelEvaluationConfig
from defaultMlProj.utils.common import read_yaml, create_directories

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, params):
        self.config = config
        self.params = params
        self.target_column = params.target_column

    def evaluate_model(self):
        try:
            logger.info("Starting model evaluation with Mlflow logging")

            os.makedirs(self.config.root_dir, exist_ok=True)

            # Load test data
            test_df = pd.read_csv(self.config.test_data_path, sep=',')
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]

            logger.info(f"Test data shape: {test_df.shape}")

            # Load trained model
            model = joblib.load(self.config.model_path)
            logger.info(f"Model loaded from {self.config.model_path}")

            # Make predictions
            y_pred = model.predict(X_test)

            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Save metrics to JSON
            metrics = {
                "r2_score": r2,
                "rmse": rmse,
                "mae": mae
            }

            with open(self.config.metric_file_name, 'w') as f:
                json.dump(metrics, f, indent=4)

            logger.info(f"Metrics saved to {self.config.metric_file_name}")

            # Set up MLflow
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment(self.config.experiment_name)

            with mlflow.start_run():
                # Log the parameters
                self.log_params_flattened(self.params.model_params)
                mlflow.log_param("target_column", self.target_column)
                mlflow.log_param("cv_splits", self.params.cv_settings.n_splits)
                
                # Log the metrics
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)

                # Saving model
                model_temp_path = Path(self.config.model_path)
                model_temp_path.parent.mkdir(parents=True, exist_ok=True)

                joblib.dump(model, model_temp_path)
                mlflow.log_artifact(model_temp_path, "model")
                logger.info(f"Model logged to MLflow as artifact: {model_temp_path}")
                

                # Log artifacts
                mlflow.log_artifact(self.config.metric_file_name)
                logger.info(f"Model and metrics logged to mlflow under experiment '{self.config.experiment_name}'")

            return metrics
        except Exception as e:
            logger.exception(f"Error occurred during model evaluation: {e}")
            raise e
        
    def log_params_flattened(self, params, parent_key=''):
        """ Recursively log parameters to MLflow, flattening nested dictionaries.
        """
        for key, value in params.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                    self.log_params_flattened(value, new_key)
            else:
                mlflow.log_param(new_key, value)

