# Configuring Dagshub for MLflow Tracking
import dagshub
import mlflow

# 🔥 Add this at the top — before any pipeline imports
dagshub.init(
    repo_owner="klan86at",
    repo_name="Default-risk-prediction",
    mlflow=True
)

# Now set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/klan86at/Default-risk-prediction.mlflow")

# Importing Pipeline libraries & logger
from defaultMlProj.pipeline.stage_data_ingestion import DataIngestionPipeline
from defaultMlProj.pipeline.stage_data_transformation import DataTransformationPipeline
from defaultMlProj.pipeline.stage_model_trainer import ModelTrainerPipeline
from defaultMlProj.pipeline.stage_model_evaluation import ModelEvaluationPipeline
from defaultMlProj import logger


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"{'>'*10} Stage: {STAGE_NAME} started {'<'*10}")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f"{'>'*10}> Stage: {STAGE_NAME} completed {'<'*10}\n\n{'X'*20}")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f"{'>'*10} Stage: {STAGE_NAME} started {'<'*10}")
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.main()
    logger.info(f"{'>'*10}> Stage: {STAGE_NAME} completed {'<'*10}\n\n{'X'*20}")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Trainer Stage"
try:
    logger.info(f"{'>'*10} Stage: {STAGE_NAME} started {'<'*10}")
    model_trainer_pipeline = ModelTrainerPipeline()
    model_trainer_pipeline.main()
    logger.info(f"{'>'*10}> Stage: {STAGE_NAME} completed {'<'*10}\n\n{'X'*20}")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f"{'>'*10} Stage: {STAGE_NAME} started {'<'*10}")
    model_evaluation_pipeline = ModelEvaluationPipeline()
    model_evaluation_pipeline.main()
    logger.info(f"{'>'*10}> Stage: {STAGE_NAME} completed {'<'*10}\n\n{'X'*20}")
except Exception as e:
    logger.exception(e)
    raise e