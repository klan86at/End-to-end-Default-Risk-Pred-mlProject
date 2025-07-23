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