from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.logging.logger import logging
from TelecoCustomerChurn.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

def evaluate_classification_model(y_true, y_pred, y_proba) -> ClassificationMetricArtifact:
    """
    Evaluates the classification model using various metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like): Predicted probabilities.

    Returns:
        ClassificationMetricArtifact: An object containing various classification metrics.
    """
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_proba)

        # Create and return the ClassificationMetricArtifact
        metric_artifact = ClassificationMetricArtifact(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc
        )
        
        return metric_artifact

    except Exception as e:
        raise CustomerChurnException(e) from e