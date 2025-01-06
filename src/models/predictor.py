import numpy as np
import pandas as pd
from snowflake.ml.model import ModelVersion
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session


def load_latest_model_version(session: Session) -> ModelVersion:
    """
    最新のモデルバージョンを取得する
    """
    # registry から最新のモデルを取得
    registry = Registry(session=session)
    registered_models = registry.show_models().sort_values(
        "created_on", ascending=False
    )
    latest_model_name = registered_models["name"].values[0]

    # モデルの参照を取得
    model_ref = registry.get_model(latest_model_name)

    # 全バージョンを取得し、バージョン名（YYMMDD形式）で降順ソートして最新を選択
    versions = model_ref.show_versions().sort_values("created_on", ascending=False)
    latest_version = versions["name"].values[0]

    mv = model_ref.version(latest_version)

    return mv


def predict(features: pd.DataFrame, mv: ModelVersion) -> np.ndarray:
    """
    モデルを用いて推論を行う
    """
    # run の結果は output_feature_0, output_feature_1 の2つの列を持つデータフレーム
    pred_probas_df = mv.run(features, function_name="predict_proba")
    return pred_probas_df.output_feature_1.values
