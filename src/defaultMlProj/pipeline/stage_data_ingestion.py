from defaultMlProj.components.data_ingestion import DataIngestion
from defaultMlProj.config.configuration import ConfigurationManager
from defaultMlProj import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.copy_data_file()
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f"{'>'*10} Stage: {STAGE_NAME} started {'<'*10}")
        obj= DataIngestionPipeline()
        obj.main()
        logger.info(f"{'>'*10}> Stage: {STAGE_NAME} completed {'<'*10}\n\n{'X'*20}")
    except Exception as e:
        logger.exception(e)
        raise e