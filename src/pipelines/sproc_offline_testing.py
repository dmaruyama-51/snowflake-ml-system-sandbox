import logging
import os
import sys

from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

from src.data.loader import fetch_test_dataset
from src.models.predictor import (
    load_default_model_version,
    load_latest_model_version,
    predict_label,
    predict_proba,
)
from src.models.trainer import calc_evaluation_metrics
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
        logger.info("Starting offline testing procedure")

        # Championモデルの取得（Defalutバージョン）
        logger.info("Loading champion model (default version)")
        champion_mv = load_default_model_version(session)
        logger.info(f"Champion model version: {champion_mv.version}")

        # Challengerモデルの取得（作成日が最新のバージョン）
        logger.info("Loading challenger model (latest version)")
        challenger_mv = load_latest_model_version(session)
        logger.info(f"Challenger model version: {challenger_mv.version}")

        # テストデータの取得
        logger.info("Fetching test dataset")
        test_df = fetch_test_dataset(session, challenger_mv)
        TARGET_COL = config["data"]["target"][0]
        test_features = test_df.drop(columns=[TARGET_COL, "UID"])
        test_target = test_df[TARGET_COL]
        logger.info(f"Test dataset size: {len(test_df)} rows")

        # モデル比較
        logger.info("Starting model comparison")
        challenger_pred_proba = predict_proba(test_features, challenger_mv)
        champion_pred_proba = predict_proba(test_features, champion_mv)
        challenger_pred_label = predict_label(test_features, challenger_mv)
        champion_pred_label = predict_label(test_features, champion_mv)

        logger.info("Calculating evaluation metrics")
        challenger_scores = calc_evaluation_metrics(
            test_target, challenger_pred_label, challenger_pred_proba
        )
        champion_scores = calc_evaluation_metrics(
            test_target, champion_pred_label, champion_pred_proba
        )

        logger.info(f"Champion model scores: {champion_scores}")
        logger.info(f"Challenger model scores: {challenger_scores}")

        if challenger_scores["PR-AUC"] > champion_scores["PR-AUC"]:
            logger.info(
                f"Challenger model (PR-AUC: {challenger_scores['PR-AUC']:.4f}) is better than Champion model (PR-AUC: {champion_scores['PR-AUC']:.4f})"
            )
            logger.info("Updating default version to challenger model")

            registry = Registry(session=session)
            m = registry.get_model("random_forest")
            m.default = challenger_mv
            logger.info("Default version updated successfully")

        else:
            logger.info(
                f"Champion model (PR-AUC: {champion_scores['PR-AUC']:.4f}) is better than Challenger model (PR-AUC: {challenger_scores['PR-AUC']:.4f})"
            )
            logger.info("No action taken")

        logger.info("Offline testing completed successfully")
        return 1

    except Exception as e:
        logger.error(
            f"An error occurred during offline testing: {str(e)}", exc_info=True
        )
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
