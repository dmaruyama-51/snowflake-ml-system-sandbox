import logging
import os
import sys
from datetime import datetime

from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

from src.data.loader import fetch_dataset
from src.data.preprocessing import split_data
from src.models.trainer import calc_evaluation_metrics, train_model
from src.utils.config import load_config
from src.utils.logger import setup_logging
from src.utils.snowflake import create_session

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPORTS_DIR = os.path.join(BASE_DIR, "src")


def sproc_training(session: Session) -> int:
    try:
        setup_logging()  # ロギング設定の初期化

        config = load_config()
        target_column = config["data"]["target"]

        df = fetch_dataset(session, is_training=True)
        if df is None:
            raise ValueError("Failed to fetch dataset")
        logger.info(f"Dataset fetched successfully. Number of rows: {len(df)}")

        df_train_val, df_test = split_data(df.drop(columns=["UID"]))
        logger.info(
            f"Dataset split completed. Training/validation data: {len(df_train_val)} rows, Test data: {len(df_test)} rows"
        )

        model_pipeline, _ = train_model(
            df=df_train_val,
            n_splits=5,
            random_state=0,
            optimize_hyperparams=True,
            n_trials=10,
        )
        logger.info("Model training completed")

        # テストデータで推論・評価
        test_scores = calc_evaluation_metrics(
            y_true=df_test[target_column],
            y_pred=model_pipeline.predict(df_test.drop(columns=target_column)),
            y_pred_proba=model_pipeline.predict_proba(
                df_test.drop(columns=target_column)
            )[:, 1],
        )
        logger.info("Model evaluation completed")

        # バージョン名に時刻も追加して一意性を確保
        # 数字始まりはNGなので、v_を先頭につける ref) https://docs.snowflake.com/en/sql-reference/identifiers-syntax
        version_name = f"v_{datetime.now().strftime('%y%m%d_%H%M%S')}"

        registry = Registry(session=session)
        _ = registry.log_model(
            model=model_pipeline,
            model_name="random_forest",
            version_name=version_name,
            metrics=test_scores,
            sample_input_data=df_train_val.drop(columns=["REVENUE"]).head(
                1
            ),  # サンプル入力データを追加
        )
        logger.info("Model logging completed")

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
            "name": "TRAINING",
            "is_permanent": True,
            "stage_location": "@practice.ml.sproc",
            "packages": [
                "snowflake-snowpark-python",
                "snowflake-ml-python",
                "scikit-learn",
                "pandas",
                "numpy",
                "optuna",
            ],
            "imports": [
                (os.path.join(IMPORTS_DIR, "data"), "src.data"),
                (os.path.join(IMPORTS_DIR, "models"), "src.models"),
                (os.path.join(IMPORTS_DIR, "utils/config.py"), "src.utils.config"),
                (os.path.join(IMPORTS_DIR, "utils/logger.py"), "src.utils.logger"),
                os.path.join(IMPORTS_DIR, "config.yml"),
            ],
            "replace": True,
            "execute_as": "caller",
        }
        session.sproc.register(func=sproc_training, **sproc_config)  # type: ignore
        session.sql("ALTER PROCEDURE TRAINING() SET LOG_LEVEL = 'INFO'").collect()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()