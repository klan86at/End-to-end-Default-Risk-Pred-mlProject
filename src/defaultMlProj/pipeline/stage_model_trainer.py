from defaultMlProj.config.configuration import ConfigurationManager
from defaultMlProj.components.model_trainer import ModelTrainer
from defaultMlProj import logger


STAGE_NAME = "Model Trainer Stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        params = config.get_params()
        model_trainer = ModelTrainer(config=model_trainer_config, params=params)
        model_trainer.train_and_evaluate()

if __name__ == "__main__":
    try:
        logger.info(f"{'>'*10} Stage: {STAGE_NAME} started {'<'*10}")
        model_trainer_pipeline = ModelTrainerPipeline()
        model_trainer_pipeline.main()
        logger.info(f"{'>'*10}> Stage: {STAGE_NAME} completed {'<'*10}\n\n{'X'*20}")
    except Exception as e:
        logger.exception(e)
        raise e