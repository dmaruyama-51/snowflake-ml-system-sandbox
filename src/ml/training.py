from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.ml.preprocessing import create_preprocessor
from src.ml.evaluation import evaluate_predictions

logger = logging.getLogger(__name__)

def create_model_pipeline(random_state: int = 0) -> Pipeline:
    """モデルパイプラインを作成"""
    logger.info("モデルパイプラインの作成を開始")
    pipeline = Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            ("classifier", RandomForestClassifier(random_state=random_state)),
        ]
    )
    logger.debug(f"パイプラインの構成: {[name for name, _ in pipeline.steps]}")
    logger.info("モデルパイプラインの作成完了")
    return pipeline

def train_model(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = 0
) -> Tuple[Pipeline, List[Dict[str, float]]]:
    """モデルの学習と交差検証を実行"""
    logger.info(f"モデルの学習を開始 (交差検証分割数: {n_splits})")
    logger.debug(f"入力データのサイズ: {df.shape}")
    
    X: pd.DataFrame = df.drop("REVENUE", axis=1)
    y: pd.Series = df["REVENUE"]
    logger.debug(f"特徴量の数: {X.shape[1]}")
    logger.debug(f"クラス分布: 正例率 {y.mean():.3f}")

    # 交差検証
    cv_scores: List[Dict[str, float]] = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Fold {fold}/{n_splits} の学習を開始")

        # データの分割
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        logger.debug(
            f"Fold {fold} データサイズ - "
            f"学習: {X_train.shape}, 検証: {X_val.shape}"
        )

        try:
            # モデルの学習と評価
            model = create_model_pipeline(random_state=random_state)
            model.fit(X_train, y_train)
            logger.debug(f"Fold {fold} のモデル学習完了")

            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            logger.info(f"Fold {fold} の評価結果:")
            metrics = evaluate_predictions(y_val, y_pred, y_pred_proba)
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
        logger.debug(f"最終モデルの特徴量重要度: {final_model.named_steps['classifier'].feature_importances_}")
    except Exception as e:
        logger.error(f"最終モデルの学習中にエラーが発生: {str(e)}")
        raise

    return final_model, cv_scores
