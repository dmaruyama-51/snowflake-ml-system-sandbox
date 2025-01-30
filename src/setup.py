import logging
from datetime import datetime
from snowflake.snowpark import Session

from src.data.dataset import create_ml_dataset
from src.data.source import prepare_online_shoppers_data
from src.utils.logger import setup_logging
from src.utils.snowflake import create_session
from src.utils.config import load_config

logger = logging.getLogger(__name__)
config = load_config()

def setup_environment(session: Session) -> None:
    try:
        setup_logging()

        database_name = session.get_current_database() or config["data"]["snowflake"]["database_dev"]
        schema_name = config["data"]["snowflake"]["schema"]
        source_table_name = config["data"]["snowflake"]["source_table"]
        dataset_table_name = config["data"]["snowflake"]["dataset_table"]

        # データベースが存在しない場合は作成
        session.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}").collect()
        logger.info(f"Ensured database {database_name} exists")

        # スキーマが存在しない場合は作成
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {database_name}.{schema_name}").collect()
        logger.info(f"Ensured schema {schema_name} exists in database {database_name}")

        # ソーステーブルを作成
        prepare_online_shoppers_data(
            session=session,
            database_name=database_name,
            schema_name=schema_name,
            table_name=source_table_name,
        )

        # データセットテーブルを作成
        today = datetime.now().strftime("%Y-%m-%d")
        create_ml_dataset(
            session=session,
            target_date=today,
            database_name=database_name,
            schema_name=schema_name,
            table_name=dataset_table_name,
            source_table_name=source_table_name,
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
        """
        ).collect()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    logger.info("Creating Snowflake session")
    session = create_session()
    setup_environment(session)

