from defaultMlProj.constants.constant import *
from defaultMlProj.utils.common import read_yaml, create_directories
from defaultMlProj.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelServingConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_data_file=config.source_data_file,
            local_data_file=config.local_data_file
        )
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
    
        create_directories([config.root_dir])
    
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
        )
    
        return model_trainer_config
    
    def get_params(self):
        """
        Returns the parameters loaded from params.yaml
        """
        return self.params
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            metric_file_name=config.metric_file_name,
            mlflow_uri=config.mlflow_uri,
            experiment_name=config.experiment_name
        )
        
        return model_evaluation_config
    
    def get_params(self):
        """
        Returns the parameters loaded from params.yaml
        """
        return self.params
    

    def get_model_serving_config(self) -> ModelServingConfig:
        config = self.config.model_serving

        create_directories([config.root_dir])

        model_serving_config = ModelServingConfig(
            model_path=config.model_path
        )

        return model_serving_config