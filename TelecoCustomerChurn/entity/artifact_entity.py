from dataclasses import dataclass

class DataIngestionArtifact:
    """
    Data Ingestion Artifact class to store the paths of the ingested data.
    """
    train_file_path: str
    test_file_path: str