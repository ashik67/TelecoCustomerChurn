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