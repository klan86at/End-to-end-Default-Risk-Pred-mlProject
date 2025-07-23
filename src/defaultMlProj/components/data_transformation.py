# components/data_transformation.py

import os
import pandas as pd
from pathlib import Path
from defaultMlProj import logger
from sklearn.model_selection import train_test_split

from defaultMlProj.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.target_column = "default_risk_score"  # actual target column name


    def train_test_split(self):
        df = pd.read_csv(self.config.data_path, sep='\t')

        try:
            logger.info("Starting data transformation: train-test split")
            logger.info(f"Full dataset shape: {df.shape}")

            # Validate target column exists
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in data. Columns: {list(df.columns)}")

            # Separate features and target
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

            logger.info(f"Feature matrix X shape: {X.shape}")  # Should be (800, 9)
            logger.info(f"Target vector y shape: {y.shape}")   # Should be (800,)

            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )

            logger.info(f"Train features shape: {X_train.shape}, Train target shape: {y_train.shape}")
            logger.info(f"Test features shape: {X_test.shape}, Test target shape: {y_test.shape}")

            # Recombine for saving (optional: keeps target in dataset)
            train_df = pd.DataFrame(X_train, columns=X.columns)
            train_df[self.target_column] = y_train.values

            test_df = pd.DataFrame(X_test, columns=X.columns)
            test_df[self.target_column] = y_test.values

            # Save to CSV
            train_csv_path = os.path.join(self.config.root_dir, "train.csv")
            test_csv_path = os.path.join(self.config.root_dir, "test.csv")

            train_df.to_csv(train_csv_path, index=False)
            test_df.to_csv(test_csv_path, index=False)

            logger.info(f"Train dataset saved to {train_csv_path}")
            logger.info(f"Test dataset saved to {test_csv_path}")

            return train_df, test_df

        except Exception as e:
            logger.exception(f"Error occurred during train-test split: {e}")
            raise e