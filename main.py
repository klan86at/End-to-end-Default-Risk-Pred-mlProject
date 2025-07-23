from defaultMlProj.pipeline.stage_data_ingestion import DataIngestionPipeline
from defaultMlProj.pipeline.stage_data_transformation import DataTransformationPipeline
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
    data_ingestion = DataTransformationPipeline()
    data_ingestion.main()
    logger.info(f"{'>'*10}> Stage: {STAGE_NAME} completed {'<'*10}\n\n{'X'*20}")
except Exception as e:
    logger.exception(e)
    raise e