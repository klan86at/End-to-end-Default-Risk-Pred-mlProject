import os
import shutil
from pathlib import Path
from defaultMlProj import logger
from defaultMlProj.utils.common import get_size

from defaultMlProj.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__ (self, config: DataIngestionConfig):
        self.config = config

    def copy_data_file(self):

        source = Path(self.config.source_data_file)
        destination = Path(self.config.local_data_file)

        try:
            logger.info(f"Starting data ingestion:copying{source} to {destination}")

            destination.parent.mkdir(parents=True, exist_ok=True)

            if not source.exists():
                raise Exception(f"Source file {source.absolute()} does not exist")

            if destination.exists():
                logger.info(f"File destination {destination} already exists. Skipping copy.")
            else:
                shutil.copy(source, destination)
                logger.info(f"File copied successfully: {source} to {destination}")

        except Exception as e:
            logger.exception(f"Error occurred while copying data file: {e}")
            raise e