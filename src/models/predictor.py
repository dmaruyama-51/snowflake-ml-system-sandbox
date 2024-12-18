import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from snowflake.ml.registry import Registry
from snowflake.ml.model import ModelVersion
from snowflake.snowpark import Session
from typing import Tuple


def load_latest_model(session: Session) -> Tuple[ModelVersion, Pipeline]:
    # registry から最新のモデルを取得
    registry = Registry(session=session)
    registered_models = registry.show_models().sort_values(
        "created_on", ascending=False
    )
    latest_model_name = registered_models["name"].values[0]

    # Registry に記録されたモデルオブジェクトではなく、モデルオブジェクトの参照が返る
    model_ref = registry.get_model(latest_model_name)
    mv = model_ref.version("v0_1_0")
    model_pipeline = mv.load(force=True)

    return mv, model_pipeline


def predict(df: pd.DataFrame, model: Pipeline) -> np.ndarray:
    if "UID" in df.columns:
        df = df.drop(columns=["UID"])
    pred_probas = model.predict_proba(df)[:, 1]
    return pred_probas
