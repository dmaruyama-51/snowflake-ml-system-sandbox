import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session


def load_latest_model(session: Session) -> Pipeline:
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

    return model_pipeline


def predict(df: pd.DataFrame, model: Pipeline) -> np.ndarray:
    pred_probas = model.predict_proba(df)[:, 1]
    return pred_probas
