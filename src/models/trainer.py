import logging
from typing import Dict, List, Optional, Tuple, Any

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
import optuna

from src.data.preprocessing import create_preprocessor
from src.utils.config import load_config

logger = logging.getLogger(__name__)
config = load_config()


def create_model_pipeline(params: Optional[Dict[str, Any]] = None, random_state: int = 0) -> Pipeline:
    """Create model pipeline with optional parameters"""
    logger.info("Starting model pipeline creation")
    
    if params is None:
        params = {}
    
    rf_params = {
        "random_state": random_state,
        **params
    }
    
    pipeline = Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            ("classifier", RandomForestClassifier(**rf_params)),
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


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, n_splits: int, random_state: int) -> float:
    """Optunaの目的関数
    
    Args:
        trial: OptunaのTrialオブジェクト
        X: 特徴量
        y: 目的変数
        n_splits: 交差検証の分割数
        random_state: 乱数シード

    Returns:
        score: PR-AUC
    """

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

        try:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model_pipeline = create_model_pipeline(params=params, random_state=random_state)
            model_pipeline.fit(X_train, y_train)
            y_pred_proba = model_pipeline.predict_proba(X_val)[:, 1]
            
            score = average_precision_score(y_val, y_pred_proba)
            cv_scores.append(score)
        except Exception as e:
            logger.error(f"Error during training fold {fold}: {str(e)}")
            raise

    avg_score = np.mean(cv_scores)
    logger.info(f"Average PR-AUC: {avg_score:.3f}")
    return avg_score


def train_model(
    df: pd.DataFrame, 
    n_splits: Optional[int] = None, 
    random_state: Optional[int] = None,
    optimize_hyperparams: bool = True,
    n_trials: int = 100
) -> Tuple[Pipeline, List[Dict[str, float]]]:
    """
    モデルの学習と交差検証を実行

    Args:
        df: 学習データ
        n_splits: 交差検証の分割数
        random_state: 乱数シード
        optimize_hyperparams: ハイパーパラメータの最適化を行うかどうか
        n_trials: Optunaの試行回数

    Returns:
        final_model_pipeline: 最終モデル
        evaluation_metrics: 評価指標
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

    if optimize_hyperparams:
        logger.info("Starting hyperparameter optimization with Optuna")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X, y, n_splits, random_state),
            n_trials=n_trials
        )
        
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best PR-AUC score: {study.best_value:.3f}")
    else:
        logger.info("Skipping hyperparameter optimization")
        best_params = {}

    # 全データで最終モデルの学習
    logger.info("Starting final model training")
    try:
        final_model_pipeline = create_model_pipeline(params=best_params, random_state=random_state)
        final_model_pipeline.fit(X, y)
        logger.info("Final model training completed")

        evaluation_metrics = calc_evaluation_metrics(
            y, 
            final_model_pipeline.predict(X), 
            final_model_pipeline.predict_proba(X)[:, 1]
        )
        logger.info("Final model evaluation completed")
    except Exception as e:
        logger.error(f"Error during final model training: {str(e)}")
        raise

    return final_model_pipeline, evaluation_metrics
