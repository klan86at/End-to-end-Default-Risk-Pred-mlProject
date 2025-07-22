from defaultMlProj.pipeline.stage_data_ingestion import DataIngestionPipeline
from defaultMlProj import logger

SATGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>>> stage {SATGE_NAME} started <<<<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>> stage {SATGE_NAME} completed <<<<<<<<\n\n(X*30)")
except Exception as e:
    logger.exception(e)
    raise e