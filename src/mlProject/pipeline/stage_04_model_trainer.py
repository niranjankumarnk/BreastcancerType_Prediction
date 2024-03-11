from mlProject.components.model_trainer import ModelTrainer
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger




STAGE_NAME = "Model Trainer stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train_model_1()
        model_trainer.train_model_2()


        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e