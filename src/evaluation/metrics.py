import numpy as np
import logging
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)


def calc_evaluation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """予測結果を評価し、メトリクスを返す"""
    logger.info("予測結果の評価を開始")
    logger.debug(f"データサイズ - y_true: {len(y_true)}, y_pred: {len(y_pred)}")

    try:
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, y_pred_proba),
            "PR-AUC": average_precision_score(y_true, y_pred_proba),
        }

        logger.info("=== 評価メトリクス ===")
        for name, score in metrics.items():
            logger.info(f"{name}: {score:.3f}")

        logger.info("評価完了")
        return metrics

    except Exception as e:
        logger.error(f"評価中にエラーが発生: {str(e)}")
        raise
