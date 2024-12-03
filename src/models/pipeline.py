import logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.data.preprocessing import create_preprocessor

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
