from defaultMlProj.components.data_ingestion import DataIngestion
from defaultMlProj.config.configuration import ConfigurationManager
from defaultMlProj.components.data_transformation import DataTransformation
from defaultMlProj import logger

STAGE_NAME = "Data Transformation Stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.train_test_split()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f"{'>'*10} Stage: {STAGE_NAME} started {'<'*10}")
        data_transformation_pipeline = DataTransformationPipeline()
        data_transformation_pipeline.main()
        logger.info(f"{'>'*10}> Stage: {STAGE_NAME} completed {'<'*10}\n\n{'X'*20}")
    except Exception as e:
        logger.exception(e)
        raise e
    