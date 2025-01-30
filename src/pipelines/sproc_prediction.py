import logging
import os
import sys

from snowflake.snowpark import Session

from src.data.loader import fetch_dataset
from src.models.predictor import load_latest_model_version, predict
from src.utils.logger import setup_logging
from src.utils.snowflake import create_session, upload_dataframe_to_snowflake
from src.utils.config import load_config

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPORTS_DIR = os.path.join(BASE_DIR, "src")


def sproc_prediction(session: Session, prediction_date: str = "2024-10-01") -> int:
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
    try:
        setup_logging()  # ロギング設定の初期化

        logger.info(f"Starting prediction process, prediction_date={prediction_date}")

        df = fetch_dataset(session, is_training=False, prediction_date=prediction_date)
        if df is None:
            raise ValueError("Failed to fetch dataset")
        features = df.drop(columns=["UID"])
        logger.info(f"Dataset fetched successfully. Number of rows: {len(df)}")

        mv = load_latest_model_version(session)
        logger.info("Model loading completed")

        df["SCORE"] = predict(features, mv)
        logger.info("Prediction completed")

        # 推論結果をスコアテーブルに書き込み
        scores_df = df[["UID", "SCORE"]]
        scores_df["SESSION_DATE"] = prediction_date
        scores_df["MODEL_NAME"] = str(mv._model_name)
        scores_df["MODEL_VERSION"] = str(mv._version_name)
        scores_df = scores_df[
            ["UID", "SESSION_DATE", "MODEL_NAME", "MODEL_VERSION", "SCORE"]
        ]

        config = load_config()
        upload_dataframe_to_snowflake(
            session=session,
            df=scores_df,
            database_name=session.get_current_database() or config["data"]["snowflake"]["database_dev"],
            schema_name=config["data"]["snowflake"]["schema"],
            table_name="SCORES",
            mode="append",
        )
        logger.info("Prediction results upload completed")
        return 1

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


if __name__ == "__main__":
    try:
        session = create_session()
        if session is None:  # セッションがNoneの場合のチェックを追加
            raise RuntimeError("Failed to create Snowflake session")

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
        session.sql(
            "ALTER PROCEDURE PREDICTION(VARCHAR) SET LOG_LEVEL = 'INFO'"
        ).collect()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()
