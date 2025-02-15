import logging
import os
import sys

from snowflake.snowpark import Session

from snowflake.ml.registry import Registry

from src.data.loader import fetch_test_dataset
from src.models.trainer import calc_evaluation_metrics
from src.models.predictor import load_default_model_version, load_latest_model_version, predict_proba, predict_label
from src.utils.config import load_config
from src.utils.logger import setup_logging
from src.utils.snowflake import create_session

logger = logging.getLogger(__name__)

config = load_config()
DATABASE_DEV = config["data"]["snowflake"]["database_dev"]
SCHEMA = config["data"]["snowflake"]["schema"]
DATASET = config["data"]["snowflake"]["dataset_table"]
SOURCE = config["data"]["snowflake"]["source_table"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPORTS_DIR = os.path.join(BASE_DIR, "src")


def sproc_offline_testing(session: Session) -> int:
    """
    Challenger vs Champion モデルの評価

    Args:
        session (Session): Snowflakeセッション

    Returns:
        int: 成功時は1、失敗時は例外を発生

    Raises:
        Exception: 処理中にエラーが発生した場合
    """
    try:
        setup_logging()

        # Championモデルの取得（Defalutバージョン）
        champion_mv = load_default_model_version(session)

        # Challengerモデルの取得（作成日が最新のバージョン）
        challenger_mv = load_latest_model_version(session)

        # テストデータの取得
        test_df = fetch_test_dataset(session, challenger_mv)
        TARGET_COL = config["data"]["target"][0]
        test_features = test_df.drop(columns=[TARGET_COL, "UID"])
        test_target = test_df[TARGET_COL]

        # モデル比較
        challenger_pred_proba = predict_proba(test_features, challenger_mv)
        champion_pred_proba = predict_proba(test_features, champion_mv)
        challenger_pred_label = predict_label(test_features, challenger_mv)
        champion_pred_label = predict_label(test_features, champion_mv)
        challenger_scores = calc_evaluation_metrics(test_target, challenger_pred_label, challenger_pred_proba)
        champion_scores = calc_evaluation_metrics(test_target, champion_pred_label, champion_pred_proba)

        if challenger_scores["PR-AUC"] > champion_scores["PR-AUC"]:
            logger.info("Challenger model is better than Champion model. Updating default version.")

            # default version を challenger に変更
            registry = Registry(session=session)
            m = registry.get_model("random_forest")
            m.default = challenger_mv

        else:
            logger.info("Champion model is better than Challenger model. No action is taken.")

        return 1

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


if __name__ == "__main__":
    try:
        session = create_session()
        if session is None:
            raise RuntimeError("Failed to create Snowflake session")

        stage_location = f"@{session.get_current_database()}.{SCHEMA}.sproc"
        sproc_config = {
            "name": "OFFLINE_TESTING",
            "is_permanent": True,
            "stage_location": stage_location,
            "packages": [
                "snowflake-snowpark-python",
                "snowflake-ml-python",
                "scikit-learn",
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
        session.sproc.register(func=sproc_offline_testing, **sproc_config)  # type: ignore
        session.sql(
            "ALTER PROCEDURE OFFLINE_TESTING() SET LOG_LEVEL = 'INFO'"
        ).collect()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()
