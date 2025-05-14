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
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBClassifier


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
                # 'LogisticRegression': LogisticRegression(verbose=1),
                'RandomForestClassifier': RandomForestClassifier(verbose=1),
                'GradientBoostingClassifier': GradientBoostingClassifier(verbose=1),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            }
            params = {
                'RandomForestClassifier': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2', None],
                },
                'GradientBoostingClassifier': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0],
                },
                'AdaBoostClassifier': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1.0],
                },
                'DecisionTreeClassifier': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'criterion': ['gini', 'entropy'],
                },
                'XGBClassifier': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                }
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
            import mlflow
            mlflow.set_experiment("TelecoCustomerChurn_Model_Training")
            with mlflow.start_run(run_name="model_training") as run:
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
                model_save_path = os.path.join(self.model_training_config.model_dir, os.path.basename(self.model_training_config.model_file_path))
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
                    'test_metrics': convert_numpy_types(asdict(test_metrics)),
                }
                logging.info(f"Saving summary report to: {report_file_path}")
                save_yaml_file(report_file_path, report_content)
                # MLflow logging
                mlflow.log_param("best_model", model_report['best_model'])
                for param_name, param_value in model_report['best_model_params'].items():
                    mlflow.log_param(param_name, param_value)
                for metric_name, metric_value in model_report['best_model_metrics'].items():
                    # Only log scalar metrics, skip dict/list (e.g., confusion_matrix)
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, float(metric_value))
                # Log train and test metrics (skip confusion_matrix)
                for k, v in convert_numpy_types(asdict(train_metrics)).items():
                    if k == "confusion_matrix":
                        continue  # Do not log confusion_matrix values as metrics
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"train_{k}", float(v))
                for k, v in convert_numpy_types(asdict(test_metrics)).items():
                    if k == "confusion_matrix":
                        continue  # Do not log confusion_matrix values as metrics
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"test_{k}", float(v))
                # Plot and log confusion matrix for train set
                if hasattr(train_metrics, 'confusion_matrix') and train_metrics.confusion_matrix is not None:
                    cm_train = np.array(train_metrics.confusion_matrix)
                    plt.figure(figsize=(5,4))
                    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
                    plt.title('Train Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    cm_train_image_path = os.path.join(self.model_training_config.report_dir, "train_confusion_matrix.png")
                    plt.savefig(cm_train_image_path)
                    plt.close()
                    mlflow.log_artifact(cm_train_image_path, artifact_path="plots")
                # Plot and log confusion matrix for test set
                if hasattr(test_metrics, 'confusion_matrix') and test_metrics.confusion_matrix is not None:
                    cm_test = np.array(test_metrics.confusion_matrix)
                    plt.figure(figsize=(5,4))
                    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
                    plt.title('Test Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    cm_test_image_path = os.path.join(self.model_training_config.report_dir, "test_confusion_matrix.png")
                    plt.savefig(cm_test_image_path)
                    plt.close()
                    mlflow.log_artifact(cm_test_image_path, artifact_path="plots")
                # Log artifacts (model, metrics.yaml, report.yaml)
                mlflow.log_artifact(model_save_path, artifact_path="model")
                mlflow.log_artifact(metrics_file_path, artifact_path="metrics")
                mlflow.log_artifact(report_file_path, artifact_path="report")
                # Log model to MLflow using the appropriate flavor
                # Use model_report to get best_model_name
                best_model_name = model_report['best_model']
                if best_model_name == 'XGBClassifier':
                    import mlflow.xgboost
                    mlflow.xgboost.log_model(best_model, artifact_path="model_mlflow")
                elif best_model_name in ['RandomForestClassifier', 'AdaBoostClassifier', 'DecisionTreeClassifier']:
                    import mlflow.sklearn
                    mlflow.sklearn.log_model(best_model, artifact_path="model_mlflow")
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