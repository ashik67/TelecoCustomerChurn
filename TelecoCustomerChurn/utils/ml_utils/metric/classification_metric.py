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
    confusion_matrix,  # <-- add this import
)

def evaluate_classification_model(y_true, y_pred, y_proba) -> ClassificationMetricArtifact:
    """
    Evaluates the classification model using various metrics, including confusion matrix.

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
        cm = confusion_matrix(y_true, y_pred)

        # Create and return the ClassificationMetricArtifact
        metric_artifact = ClassificationMetricArtifact(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm.tolist()  # ensure it's serializable
        )
        
        return metric_artifact

    except Exception as e:
        import sys
        raise CustomerChurnException(e, sys) from e