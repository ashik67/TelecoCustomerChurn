from TelecoCustomerChurn.entity.config_entity import ModelPusherConfig
from TelecoCustomerChurn.entity.artifact_entity import ModelTrainingArtifact, DataTransformationArtifact, ModelPusherArtifact
from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.logging.logger import logging
from TelecoCustomerChurn.utils.main_utils import load_object, save_object
import os
import sys
import shutil

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_training_artifact: ModelTrainingArtifact, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_training_artifact = model_training_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def push_model(self) -> ModelPusherArtifact:
        try:
            os.makedirs(self.model_pusher_config.final_model_dir, exist_ok=True)
            # Load model and preprocessor from artifacts
            model = load_object(self.model_training_artifact.model_file_path)
            preprocessor = load_object(self.data_transformation_artifact.preprocessed_object_file_path)
            # Define destination paths
            pushed_model_path = os.path.join(self.model_pusher_config.final_model_dir, 'model.pkl')
            pushed_preprocessor_path = os.path.join(self.model_pusher_config.final_model_dir, 'preprocessed_object.pkl')
            # Save/copy model and preprocessor
            save_object(pushed_model_path, model)
            save_object(pushed_preprocessor_path, preprocessor)
            logging.info(f"Model and preprocessor pushed to {self.model_pusher_config.final_model_dir}")
            return ModelPusherArtifact(
                final_model_dir=self.model_pusher_config.final_model_dir,
                pushed_model_path=pushed_model_path,
                pushed_preprocessor_path=pushed_preprocessor_path
            )
        except Exception as e:
            logging.error(f"Error in ModelPusher: {e}")
            raise CustomerChurnException(e, sys) from e
