import logging
import os
import sys

from snowflake.snowpark import Session

from src.data.dataset import update_ml_dataset
from src.utils.logger import setup_logging
from src.utils.snowflake import create_session

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPORTS_DIR = os.path.join(BASE_DIR, "src")


def sproc_dataset(session: Session, target_date: str) -> int:
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
        setup_logging()
        
        # データベース名とスキーマ名を取得し、Noneの場合はデフォルト値を使用
        database_name = session.get_current_database() or "practice"
        schema_name = session.get_current_schema() or "ml"
        
        update_ml_dataset(
            session=session, 
            target_date=target_date,
            database_name=database_name,
            schema_name=schema_name,
            table_name="dataset",
            source_table_name="online_shoppers_intention",
        )
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
            "name": "DATASET",
            "is_permanent": True,
            "stage_location": "@practice.ml.sproc",
            "packages": [
                "snowflake-snowpark-python"
            ],
            "imports": [
                (os.path.join(IMPORTS_DIR, "data"), "src.data"),
                (os.path.join(IMPORTS_DIR, "utils/logger.py"), "src.utils.logger"),
                (
                    os.path.join(IMPORTS_DIR, "utils/snowflake.py"),
                    "src.utils.snowflake",
                ),
            ],
            "replace": True,
            "execute_as": "caller",
        }
        session.sproc.register(func=sproc_dataset, **sproc_config)  # type: ignore
        session.sql(
            "ALTER PROCEDURE DATASET(VARCHAR) SET LOG_LEVEL = 'INFO'"
        ).collect()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()
