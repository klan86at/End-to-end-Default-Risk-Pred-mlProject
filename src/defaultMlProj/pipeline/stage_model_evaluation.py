from defaultMlProj.config.configuration import ConfigurationManager
from defaultMlProj.components.model_evaluation import ModelEvaluation
from defaultMlProj import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_eval_config = config.get_model_evaluation_config()
        params = config.get_params()
        model_eval = ModelEvaluation(config=model_eval_config, params=params)
        metrics = model_eval.evaluate_model()
        logger.info(f"Model evaluation completed with metrics: {metrics}")


if __name__ == "__main__":
    try:
        logger.info(f"{'>'*10} Stage: {STAGE_NAME} started {'<'*10}")
        model_evaluation_pipeline = ModelEvaluationPipeline()
        model_evaluation_pipeline.main()
        logger.info(f"{'>'*10}> Stage: {STAGE_NAME} completed {'<'*10}\n\n{'X'*20}")
    except Exception as e:
        logger.exception(e)
        raise e