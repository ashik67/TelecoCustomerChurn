from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Data Ingestion Artifact class to store the paths of the ingested data.
    """
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    """
    Data Validation Artifact class to store the paths of the valid and invalid data.
    """
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    """
    Data Transformation Artifact class to store the paths of the transformed data.
    """
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_train_target_file_path: str
    transformed_test_target_file_path: str
    preprocessed_object_file_path: str

@dataclass
class ClassificationMetricArtifact:
    """
    Classification Metric Artifact class to store evaluation metrics for classification models.
    """
    accuracy: float                     # Accuracy of the model
    precision: float                    # Precision of the model
    recall: float                       # Recall of the model
    f1_score: float                     # F1 score of the model
    roc_auc: float                      # ROC AUC score of the model
    confusion_matrix: list              # Confusion matrix as a list of lists

@dataclass
class ModelTrainingArtifact:
    """
    Model Training Artifact class to store the paths and metadata of the trained model and its outputs.
    """
    model_file_path: str                # Path to the serialized/trained model (e.g., model.pkl)
    metrics_file_path: str              # Path to the file containing evaluation metrics (e.g., metrics.yaml)
    report_file_path: str               # Path to the human-readable training report (e.g., report.yaml)
    model_dir: str                      # Directory where the model is stored
    metrics_dir: str                    # Directory for storing all metrics files
    report_dir: str                     # Directory for storing all reports
    expected_score: float               # The expected/target score for the model
    fitting_thresholds: dict            # Overfitting/underfitting thresholds used for model selection or alerts
    training_timestamp: str             # Timestamp of when the model was trained
    train_metric_artifact: ClassificationMetricArtifact  # Metrics for the training dataset
    test_metric_artifact: ClassificationMetricArtifact   # Metrics for the testing dataset