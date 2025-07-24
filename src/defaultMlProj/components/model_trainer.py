import os
import pandas as pd
import numpy as np
import joblib
import xgboost
from defaultMlProj import logger
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import StackingRegressor    
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from defaultMlProj.entity.config_entity import ModelTrainerConfig
from defaultMlProj.utils.common import read_yaml, create_directories

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, params):
        self.config = config
        self.params = params
        self.target_column = params.target_column

    def create_model(self):
        """ Creates a dictionary of models based on the parameters provided.
        Each model is wrapped in a Pipeline with a StandardScaler.
        The models include:
        - Linear Regression
        - KNN
        - Decision Tree
        - Random Forest
        - Stacking Regressor (using the above models as base estimators)
        The final estimator for the Stacking Regressor is a Linear Regression model.
        The models are returned as a dictionary where keys are model names and values are the model instances.
        Returns:
            dict: A dictionary where keys are model names and values are the model instances.
        """
        logger.info("Creating models based on provided parameters")
        try:
            # Extract params
            p = self.params.model_params

            models = {}

            # Linear Regression
            models['LinearRegression'] = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression(
                    fit_intercept=p.linear_regression.fit_intercept
                ))
            ])

            # KNN
            models['KNN'] = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', KNeighborsRegressor(
                    n_neighbors=p.knn.n_neighbors,
                    weights=p.knn.weights,
                    algorithm=p.knn.algorithm
                ))
            ])

            # Decision Tree
            models['DecisionTree'] = DecisionTreeRegressor(
                criterion=p.decision_tree.criterion,
                max_depth=p.decision_tree.max_depth,
                min_samples_split=p.decision_tree.min_samples_split,
                min_samples_leaf=p.decision_tree.min_samples_leaf,
                random_state=p.decision_tree.random_state
            )
            
            # Random Forest
            models['RandomForest'] = RandomForestRegressor(
                n_estimators=p.random_forest.n_estimators,
                criterion=p.random_forest.criterion,
                max_depth=p.random_forest.max_depth,
                min_samples_split=p.random_forest.min_samples_split,
                min_samples_leaf=p.random_forest.min_samples_leaf,
                random_state=p.random_forest.random_state
            )

            # Stacking Regressor
            base_estimators = list(models.items())

            final_estimator = LinearRegression(
                fit_intercept=p.linear_regression.fit_intercept
            )

            stacking = StackingRegressor(
                estimators=base_estimators,
                final_estimator=final_estimator,
                cv=p.stacking_regressor.cv,
                n_jobs=p.stacking_regressor.n_jobs
            )

            models['Stacking Regressor'] = stacking
            logger.info(f"Models created: {list(models.keys())}")
            return models
        
        except Exception as e:
            logger.exception(f"Error occurred while creating models: {e}")
            raise e
        
    def train_and_evaluate(self):
        logger.info("Starting model training with external parameters")
        try:
            # Load data
            train_df = pd.read_csv(self.config.train_data_path, sep=',')
            test_df = pd.read_csv(self.config.test_data_path, sep=',')
            
            logger.info(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")

            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]

            logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # Create models using params
            models = self.create_model()

            # Get CV settings from params
            cv_params = self.params.cv_settings
            cv = KFold(
                n_splits=cv_params.n_splits,
                shuffle=cv_params.shuffle,
                random_state=cv_params.random_state
            )
            results = {}

            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                    results[name] = scores
                    logger.info(f"{name} R2 = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                except Exception as e:
                    logger.exception(f"Failed to evaluate {name}: {e}")
                    raise e
                
            # The best model
            best_name = max(results, key=lambda k: results[k].mean())
            best_model = models[best_name].fit(X_train, y_train)

            # Final evaluation
            y_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            logger.info(f"Best model: {best_name} | Test R2 : {test_r2:.4f}, RMSE : {test_rmse:.4f}")

            # Save model
            Path(self.config.model_name).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, self.config.model_name)
            logger.info(f"Model saved to {self.config.model_name}")

            return best_model, test_r2, test_rmse
        except Exception as e:
            logger.exception(f"Error occurred during model training and evaluation: {e}")
            raise e