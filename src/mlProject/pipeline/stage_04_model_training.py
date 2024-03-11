from mlProject.components.model_training import ModelTraining
from mlProject.entity.config_entity import ModelTrainingConfig
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger




STAGE_NAME = "Model Training stage"



class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config= model_training_config)
        model_training.train_model()



        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e