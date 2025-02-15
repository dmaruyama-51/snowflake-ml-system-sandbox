import numpy as np
import pandas as pd
from snowflake.ml.model import ModelVersion
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session


def load_latest_model_version(session: Session) -> ModelVersion:
    """
    最新のモデルバージョンを取得する
    """
    registry = Registry(session=session)
    model_ref = registry.get_model("random_forest")
    mv = model_ref.last()

    return mv


def load_default_model_version(session: Session) -> ModelVersion:
    """
    デフォルトバージョンを取得する
    """
    registry = Registry(session=session)
    model_ref = registry.get_model("random_forest")
    mv = model_ref.default
    return mv


def predict_proba(features: pd.DataFrame, mv: ModelVersion) -> np.ndarray:
    """
    モデルを用いて推論を行う
    """
    # run の結果は output_feature_0, output_feature_1 の2つの列を持つデータフレーム
    pred_probas_df = mv.run(features, function_name="predict_proba")
    return pred_probas_df.output_feature_1.values

def predict_label(features: pd.DataFrame, mv: ModelVersion) -> np.ndarray:
    """
    モデルを用いて推論を行う
    """
    pred_df = mv.run(features, function_name="predict")
    return pred_df.output_feature_0.values
