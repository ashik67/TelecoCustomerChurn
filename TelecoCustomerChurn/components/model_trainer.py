from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.logging.logger import logging
from TelecoCustomerChurn.entity.config_entity import ModelTrainingConfig
from TelecoCustomerChurn.entity.artifact_entity import DataTransformationArtifact, ModelTrainingArtifact
from TelecoCustomerChurn.utils.main_utils import load_numpy_array_data, save_object,evaluate_models,load_object
from TelecoCustomerChurn.utils.ml_utils.metric.classification_metric import evaluate_classification_model
from TelecoCustomerChurn.utils.ml_utils.model.estimator import TelecoCustomerChurnModel
from TelecoCustomerChurn.utils.main_utils import save_yaml_file, convert_numpy_types
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report,r2_score
from sklearn.model_selection import GridSearchCV
from dataclasses import asdict


class ModelTrainer:
    def __init__(self, model_training_config: ModelTrainingConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info("Initializing ModelTrainer with provided config and data transformation artifact.")
            self.model_training_config = model_training_config
            self.data_transformation_artifact = data_transformation_artifact
            logging.info(f"ModelTrainingConfig: {self.model_training_config}")
            logging.info(f"DataTransformationArtifact: {self.data_transformation_artifact}")
        except Exception as e:
            logging.error(f"Error during ModelTrainer initialization: {e}")
            raise CustomerChurnException(e, sys) from e
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Train multiple models, select the best one based on accuracy, and return the best trained model and the model report.
        """
        try:
            logging.info("Setting up models and hyperparameter grids for training.")
            models = {
                #'LogisticRegression': LogisticRegression(verbose=1),
                'RandomForestClassifier': RandomForestClassifier(verbose=1),
                #'GradientBoostingClassifier': GradientBoostingClassifier(verbose=1),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
            }
            params = {
                #'LogisticRegression': {},
                'RandomForestClassifier': {'n_estimators': [50, 100, 200]},
                #'GradientBoostingClassifier': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.8, 1.0]},
                'AdaBoostClassifier': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5, 1.0]},
                'DecisionTreeClassifier': {'criterion': ['gini', 'entropy', 'log_loss']}
            }
            logging.info("Beginning model evaluation and hyperparameter tuning.")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            best_model_name = model_report['best_model']
            best_model_params = model_report['best_model_params']
            logging.info(f"Best model selected: {best_model_name} with params: {best_model_params}")
            # Refit the best model on the full training data with best params
            best_model = models[best_model_name].set_params(**best_model_params)
            logging.info(f"Fitting the best model: {best_model_name} on the training data.")
            best_model.fit(X_train, y_train)
            logging.info(f"Model {best_model_name} training complete.")
            return best_model, model_report
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise CustomerChurnException(e, sys) from e

    def initiate_model_training(self) -> ModelTrainingArtifact:
        try:
            logging.info("Starting model training process.")
            # Load the preprocessed train and test data
            logging.info(f"Loading preprocessed train data from: {self.data_transformation_artifact.transformed_train_file_path}")
            X_train = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            y_train = load_numpy_array_data(self.data_transformation_artifact.transformed_train_target_file_path)
            logging.info(f"Loading preprocessed test data from: {self.data_transformation_artifact.transformed_test_file_path}")   
            X_test = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            y_test = load_numpy_array_data(self.data_transformation_artifact.transformed_test_target_file_path)
            # Check if the data is loaded correctly
            if X_train is None or y_train is None or X_test is None or y_test is None:
                logging.error("Failed to load the preprocessed data.")
                raise ValueError("Failed to load the preprocessed data.")
            # Check if the data is empty
            if X_train.size == 0 or y_train.size == 0 or X_test.size == 0 or y_test.size == 0:
                logging.error("The preprocessed data is empty.")
                raise ValueError("The preprocessed data is empty.")
            logging.info(f"Train data shape: {X_train.shape}, Train target shape: {y_train.shape}")
            logging.info(f"Test data shape: {X_test.shape}, Test target shape: {y_test.shape}")
            # model training
            logging.info("Training the model.")
            best_model, model_report = self.train_model(X_train, y_train, X_test, y_test)
            # Ensure all output directories exist before saving files
            logging.info("Ensuring all output directories exist for model, metrics, and report.")
            os.makedirs(self.model_training_config.model_dir, exist_ok=True)
            os.makedirs(self.model_training_config.metrics_dir, exist_ok=True)
            os.makedirs(self.model_training_config.report_dir, exist_ok=True)
            # Evaluate metrics for best model
            from TelecoCustomerChurn.utils.ml_utils.metric.classification_metric import evaluate_classification_model
            logging.info("Generating predictions and probabilities for train and test sets.")
            train_pred = best_model.predict(X_train)
            test_pred = best_model.predict(X_test)
            train_proba = best_model.predict_proba(X_train)[:, 1] if hasattr(best_model, 'predict_proba') else None
            test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            logging.info("Evaluating classification metrics for train and test sets.")
            train_metrics = evaluate_classification_model(y_train, train_pred, train_proba)
            test_metrics = evaluate_classification_model(y_test, test_pred, test_proba)
            # Save the best model with preprocessor
            logging.info(f"Loading preprocessor object from: {self.data_transformation_artifact.preprocessed_object_file_path}")
            preprocessed_object = load_object(self.data_transformation_artifact.preprocessed_object_file_path)
            from TelecoCustomerChurn.utils.ml_utils.model.estimator import TelecoCustomerChurnModel
            churn_model = TelecoCustomerChurnModel(preprocessed_object=preprocessed_object, model=best_model)
            model_save_path = os.path.join(self.model_training_config.model_dir, self.model_training_config.model_file_path)
            logging.info(f"Saving trained model to: {model_save_path}")
            save_object(file_path=model_save_path, obj=churn_model)
            # Save metrics to metrics.yaml in metrics_dir
            metrics_file_path = os.path.join(self.model_training_config.metrics_dir, os.path.basename(self.model_training_config.metrics_file_path))
            logging.info(f"Saving model report to metrics file: {metrics_file_path}")
            clean_model_report = convert_numpy_types(model_report)
            save_yaml_file(metrics_file_path, clean_model_report)
            # Save a simple report to report.yaml in report_dir
            report_file_path = os.path.join(self.model_training_config.report_dir, os.path.basename(self.model_training_config.report_file_path))
            report_content = {
                'best_model': model_report['best_model'],
                'best_model_metrics': convert_numpy_types(model_report['best_model_metrics']),
                'train_metrics': convert_numpy_types(asdict(train_metrics)),
                'test_metrics': convert_numpy_types(asdict(test_metrics))
            }
            logging.info(f"Saving summary report to: {report_file_path}")
            save_yaml_file(report_file_path, report_content)
            # Prepare ModelTrainerArtifact (update as needed for your artifact_entity.py)
            from TelecoCustomerChurn.entity.artifact_entity import ModelTrainingArtifact, ClassificationMetricArtifact
            logging.info("Creating metric artifacts for train and test sets.")
            # train_metrics and test_metrics are already ClassificationMetricArtifact objects
            train_metric_artifact = train_metrics
            test_metric_artifact = test_metrics
            artifact = ModelTrainingArtifact(
                model_file_path=model_save_path,
                metrics_file_path=metrics_file_path,
                report_file_path=report_file_path,
                model_dir=self.model_training_config.model_dir,
                metrics_dir=self.model_training_config.metrics_dir,
                report_dir=self.model_training_config.report_dir,
                expected_score=self.model_training_config.expected_score,
                fitting_thresholds=self.model_training_config.fitting_thresholds,
                training_timestamp=self.model_training_config.training_timestamp,
                train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact
            )
            logging.info(f"Model training pipeline completed successfully. ModelTrainerArtifact: {artifact}")
            return artifact
        except Exception as e:
            logging.error(f"Error during model training pipeline: {e}")
            raise CustomerChurnException(e, sys) from e