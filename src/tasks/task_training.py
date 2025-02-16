import logging
import sys

from snowflake.snowpark import Session

from src.utils.logger import setup_logging
from src.utils.snowflake import create_session

logger = logging.getLogger(__name__)


def create_training_task(session: Session) -> None:
    """
    トレーニング用のストアドプロシージャを実行するタスクを作成する

    Args:
        session (Session): Snowflakeセッション

    Raises:
        Exception: タスクの作成に失敗した場合
    """
    try:
        setup_logging()
        logger.info("Starting training task creation")

        # タスクの作成
        create_task_sql = """
        CREATE OR REPLACE TASK task_training
            WAREHOUSE = COMPUTE_WH
            SCHEDULE = 'USING CRON 0 10 1 * * Asia/Tokyo'  -- 毎月1日の午前10時に実行
        AS
            CALL training();
        """
        session.sql(create_task_sql).collect()
        logger.info("Task created successfully")

        # タスクの有効化
        session.sql("ALTER TASK task_training RESUME").collect()
        logger.info("Task resumed successfully")

    except Exception as e:
        error_msg = f"Failed to create training task: {str(e)}"
        logger.error(error_msg)
        raise


if __name__ == "__main__":
    try:
        session = create_session()
        if session is None:
            raise RuntimeError("Failed to create Snowflake session")

        create_training_task(session)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        if session:
            session.close()
