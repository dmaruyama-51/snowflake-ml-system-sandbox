import logging
from datetime import datetime

from snowflake.snowpark import Session

from src.data.dataset import create_ml_dataset
from src.data.source import prepare_online_shoppers_data
from src.utils.constants import DATABASE_DEV, DATASET, SCHEMA, SOURCE
from src.utils.logger import setup_logging
from src.utils.snowflake import create_session

logger = logging.getLogger(__name__)


def setup_environment(session: Session) -> None:
    try:
        setup_logging()

        database_name = session.get_current_database() or DATABASE_DEV

        # データベースが存在しない場合は作成
        session.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}").collect()
        logger.info(f"Ensured database {database_name} exists")

        # スキーマが存在しない場合は作成
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {database_name}.{SCHEMA}").collect()
        logger.info(f"Ensured schema {SCHEMA} exists in database {database_name}")

        # ソーステーブルを作成
        prepare_online_shoppers_data(
            session=session,
            database_name=database_name,
            schema_name=SCHEMA,
            table_name=SOURCE,
        )

        # データセットテーブルを作成
        today = datetime.now().strftime("%Y-%m-%d")
        create_ml_dataset(
            session=session,
            target_date=today,
            database_name=database_name,
            schema_name=SCHEMA,
            table_name=DATASET,
            source_table_name=SOURCE,
        )

        # Scores テーブルを作成
        session.sql("""
            create or replace table SCORES (
                UID VARCHAR(16777216) NOT NULL,
                SESSION_DATE DATE NOT NULL,
                MODEL_NAME VARCHAR(16777216),
                MODEL_VERSION VARCHAR(16777216),
                SCORE FLOAT,
                primary key (UID, SESSION_DATE)
            )        
        """).collect()
        logger.info("Created SCORES table")

        # sproc ステージを作成
        session.sql("""
            CREATE STAGE IF NOT EXISTS sproc
            DIRECTORY = (ENABLE = TRUE)
        """).collect()
        logger.info("Created sproc stage")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


if __name__ == "__main__":
    session = create_session()
    if session is None:
        raise ValueError("Failed to create Snowflake session")

    setup_environment(session)
