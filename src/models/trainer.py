from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from src.models.pipeline import create_model_pipeline
from src.utils.config import load_config
from src.evaluation.metrics import calc_evaluation_metrics

logger = logging.getLogger(__name__)
config = load_config()


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
    # configから設定値を取得
    if n_splits is None:
        n_splits = config["model"]["cv"]["n_splits"]
    if random_state is None:
        random_state = config["model"]["random_forest"]["random_state"]

    target_column = config["data"]["target"]

    logger.info(f"モデルの学習を開始 (交差検証分割数: {n_splits})")
    logger.debug(f"入力データのサイズ: {df.shape}")

    X: pd.DataFrame = df.drop(target_column, axis=1)
    y: pd.Series = df[target_column]

    # 交差検証
    cv_scores: List[Dict[str, float]] = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Fold {fold}/{n_splits} の学習を開始")

        # データの分割
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        logger.debug(
            f"Fold {fold} データサイズ - 学習: {X_train.shape}, 検証: {X_val.shape}"
        )

        try:
            # モデルの学習と評価
            model = create_model_pipeline(random_state=random_state)
            model.fit(X_train, y_train)
            logger.debug(f"Fold {fold} のモデル学習完了")

            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            logger.info(f"Fold {fold} の評価結果:")
            metrics = calc_evaluation_metrics(y_val, y_pred, y_pred_proba)
            cv_scores.append(metrics)

        except Exception as e:
            logger.error(f"Fold {fold} の学習中にエラーが発生: {str(e)}")
            raise

    # 結果のサマリー
    logger.info("=== Cross-validation Summary ===")
    for metric in cv_scores[0].keys():
        scores = [fold[metric] for fold in cv_scores]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        logger.info(f"{metric}: {mean_score:.3f} (±{std_score:.3f})")

    # 最終モデルの学習
    logger.info("最終モデルの学習を開始")
    try:
        final_model = create_model_pipeline(random_state=random_state)
        final_model.fit(X, y)
        logger.info("最終モデルの学習完了")
    except Exception as e:
        logger.error(f"最終モデルの学習中にエラーが発生: {str(e)}")
        raise

    return final_model, cv_scores
