import logging
import sys
from datetime import datetime, timedelta, timezone

from snowflake.snowpark import Session

from src.utils.logger import setup_logging
from src.utils.snowflake import create_session

logger = logging.getLogger(__name__)


def create_prediction_task(session: Session) -> None:
    """
    データセットのストアドプロシージャを実行するタスクを作成

    Args:
        session (Session): Snowflakeセッション

    Raises:
        Exception: タスクの作成に失敗した場合
    """
    try:
        setup_logging()
        logger.info("Starting dataset task creation")

        jst = timezone(timedelta(hours=9))
        yesterday = (datetime.now(jst) - timedelta(days=1)).strftime("%Y-%m-%d")

        # タスクの作成
        create_task_sql = f"""
        CREATE OR REPLACE TASK task_dataset
            WAREHOUSE = COMPUTE_WH
            SCHEDULE = 'USING CRON 0 5 * * * Asia/Tokyo'
        AS
            CALL dataset('{yesterday}');
        """
        session.sql(create_task_sql).collect()
        logger.info("Task created successfully")

        # タスクの有効化
        session.sql("ALTER TASK practice.ml.task_dataset RESUME").collect()
        logger.info("Task resumed successfully")

    except Exception as e:
        error_msg = f"Failed to create prediction task: {str(e)}"
        logger.error(error_msg)
        raise


if __name__ == "__main__":
    try:
        session = create_session()
        if session is None:
            raise RuntimeError("Failed to create Snowflake session")

        create_prediction_task(session)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()
