import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.preprocessing import create_preprocessor
from src.utils.config import load_config

logger = logging.getLogger(__name__)
config = load_config()


def create_model_pipeline(random_state: int = 0) -> Pipeline:
    """Create model pipeline"""
    logger.info("Starting model pipeline creation")
    pipeline = Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            ("classifier", RandomForestClassifier(random_state=random_state)),
        ]
    )
    logger.debug(f"Pipeline components: {[name for name, _ in pipeline.steps]}")
    logger.info("Model pipeline creation completed")
    return pipeline


def calc_evaluation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """Evaluate predictions and return metrics"""
    logger.info("Starting prediction evaluation")
    logger.debug(f"Data size - y_true: {len(y_true)}, y_pred: {len(y_pred)}")

    try:
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, y_pred_proba),
            "PR-AUC": average_precision_score(y_true, y_pred_proba),
        }

        logger.info("=== Evaluation Metrics ===")
        for name, score in metrics.items():
            logger.info(f"{name}: {score:.3f}")

        logger.info("Evaluation completed")
        return metrics

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise ValueError(
            "Failed to calculate evaluation metrics. Please check input data."
        )


def train_model(
    df: pd.DataFrame, n_splits: Optional[int] = None, random_state: Optional[int] = None
) -> Tuple[Pipeline, List[Dict[str, float]]]:
    """
    モデルの学習と交差検証を実行

    Args:
        df: 学習データ
        n_splits: 交差検証の分割数（Noneの場合はconfig.ymlの値を使用）
        random_state: 乱数シード（Noneの場合はconfig.ymlの値を使用）
    """
    # configからデフォルト値を取得
    if n_splits is None:  # pragma: no cover
        n_splits = config["model"]["cv"]["n_splits"]
    if random_state is None:  # pragma: no cover
        random_state = config["model"]["random_forest"]["random_state"]

    target_column = config["data"]["target"]

    logger.info(f"Starting model training (cross-validation splits: {n_splits})")
    logger.debug(f"Input data size: {df.shape}")

    X: pd.DataFrame = df.drop(target_column, axis=1)
    y: pd.Series = df[target_column]

    # 交差検証
    cv_scores: List[Dict[str, float]] = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Starting training for fold {fold}/{n_splits}")

        # データの分割
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        logger.debug(
            f"Fold {fold} data size - train: {X_train.shape}, validation: {X_val.shape}"
        )

        try:
            # モデルの学習と評価
            model_pipeline = create_model_pipeline(random_state=random_state)
            model_pipeline.fit(X_train, y_train)
            logger.debug(f"Model training completed for fold {fold}")

            y_pred = model_pipeline.predict(X_val)
            y_pred_proba = model_pipeline.predict_proba(X_val)[:, 1]

            logger.info(f"Fold {fold} evaluation results:")
            metrics = calc_evaluation_metrics(y_val, y_pred, y_pred_proba)
            cv_scores.append(metrics)

        except Exception as e:
            logger.error(f"Error during training fold {fold}: {str(e)}")
            raise

    # 結果のサマリー
    logger.info("=== Cross-validation Summary ===")
    for metric in cv_scores[0].keys():
        scores = [fold[metric] for fold in cv_scores]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        logger.info(f"{metric}: {mean_score:.3f} (±{std_score:.3f})")

    # 最終モデルの学習
    logger.info("Starting final model training")
    try:
        final_model_pipeline = create_model_pipeline(random_state=random_state)
        final_model_pipeline.fit(X, y)
        logger.info("Final model training completed")
    except Exception as e:
        logger.error(f"Error during final model training: {str(e)}")
        raise

    return final_model_pipeline, cv_scores
