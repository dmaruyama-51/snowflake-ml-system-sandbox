import os
import sys

from snowflake.snowpark import Session

from src.data.loader import fetch_dataset
from src.models.predictor import load_latest_model, predict
from src.utils.logger import log_to_snowflake
from src.utils.snowflake import create_session, upload_dataframe_to_snowflake

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPORTS_DIR = os.path.join(BASE_DIR, "src")


def sproc_prediction(session: Session, prediction_date: str) -> int:
    """
    推論Sprocの処理内容

    Args:
        session (Session): Snowflakeセッション
        prediction_date (str): 推論日付（YYYY-MM-DD）

    Returns:
        int: 成功時は1、失敗時は例外を発生

    Raises:
        Exception: 処理中にエラーが発生した場合
    """
    log_to_snowflake(session, f"推論処理を開始, predicion_date={prediction_date}")
    try:
        df = fetch_dataset(session, is_training=False, prediction_date=prediction_date)
        if df is None:
            raise ValueError("データセットが取得できませんでした")
        log_to_snowflake(session, f"データセットのフェッチ完了。行数: {len(df)}")

        mv, model = load_latest_model(session)
        log_to_snowflake(session, "モデルの読み込み完了")
        df["SCORE"] = predict(df, model)
        log_to_snowflake(session, "推論完了")

        # 推論結果をスコアテーブルに書き込み
        scores_df = df[["UID", "SCORE"]]
        scores_df["SESSION_DATE"] = prediction_date
        scores_df["MODEL_NAME"] = str(mv._model_name)
        scores_df["MODEL_VERSION"] = str(mv._version_name)
        scores_df = scores_df[
            ["UID", "SESSION_DATE", "MODEL_NAME", "MODEL_VERSION", "SCORE"]
        ]
        upload_dataframe_to_snowflake(
            session=session,
            df=scores_df,
            database_name="PRACTICE",
            schema_name="ML",
            table_name="SCORES",
            mode="append",
        )
        log_to_snowflake(session, "推論結果のアップロード完了")
        return 1

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        session = create_session()
        sproc_config = {
            "name": "PREDICTION",
            "is_permanent": True,
            "stage_location": "@practice.ml.sproc",
            "packages": [
                "snowflake-snowpark-python",
                "snowflake-ml-python",
                "scikit-learn",
                "pandas",
                "numpy",
            ],
            "imports": [
                (os.path.join(IMPORTS_DIR, "data"), "src.data"),
                (os.path.join(IMPORTS_DIR, "models"), "src.models"),
                (os.path.join(IMPORTS_DIR, "utils/config.py"), "src.utils.config"),
                (os.path.join(IMPORTS_DIR, "utils/logger.py"), "src.utils.logger"),
                (
                    os.path.join(IMPORTS_DIR, "utils/snowflake.py"),
                    "src.utils.snowflake",
                ),
                os.path.join(IMPORTS_DIR, "config.yml"),
            ],
            "replace": True,
            "execute_as": "caller",
        }
        session.sproc.register(func=sproc_prediction, **sproc_config)  # type: ignore

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()
