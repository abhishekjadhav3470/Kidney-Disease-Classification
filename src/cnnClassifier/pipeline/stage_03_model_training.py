from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger
import os

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)

        base_model_path = "artifacts/prepare_base_model/base_model_updated.h5"
        if not os.path.exists(base_model_path):
            logger.error(f"Base model not found at {base_model_path}. Ensure the file exists.")
            return  # Exit the function or handle the error appropriately

        training.get_base_model(base_model_path)
        training.train_valid_generator()
        training.train()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
