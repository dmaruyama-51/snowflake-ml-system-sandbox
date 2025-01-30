import logging
import random
import uuid

import pandas as pd
from snowflake.snowpark.session import Session
from ucimlrepo import fetch_ucirepo

from src.utils.snowflake import upload_dataframe_to_snowflake
from src.utils.config import load_config

logger = logging.getLogger(__name__)

random.seed(0)

# MONTHカラムの値を月番号に変換する辞書
MONTH_TO_NUM = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "June": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}


def get_year(month: str) -> str:
    """月に応じて年を設定する"""
    return (
        "2025"
        if month in ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep"]
        else "2024"
    )


def get_random_day(month: str) -> str:
    """月に応じて日付をランダムに割り振る"""
    days_in_month = {
        "Apr": 30,
        "June": 30,
        "Sep": 30,
        "Nov": 30,
        "Feb": 28,  # 2024/2025年なので2月は28日
        "Jan": 31,
        "Mar": 31,
        "May": 31,
        "Jul": 31,
        "Aug": 31,
        "Oct": 31,
        "Dec": 31,
    }
    return str(random.randint(1, days_in_month[month])).zfill(2)


def prepare_online_shoppers_data(
    session: Session,
    database_name: str | None = None,
    schema_name: str | None = None,
    table_name: str | None = None,
    mode: str = "overwrite",
) -> None:
    """
    Online Shoppers Intention データセットを取得し、Snowflakeにロードする

    Args:
        session (AsyncSession): Snowflakeセッション
        database_name (str): ロード先のデータベース名
        schema_name (str): ロード先のスキーマ名
        table_name (str): ロード先のテーブル名
        mode (str, optional): データ書き込みモード. Defaults to 'overwrite'.

    Raises:
        Exception: データの取得中にエラーが発生した場合
    """
    try:
        config = load_config()
        database_name = database_name or session.get_current_database() or config["data"]["snowflake"]["database_dev"]
        schema_name = schema_name or config["data"]["snowflake"]["schema"]
        table_name = table_name or config["data"]["snowflake"]["source_table"]

        logger.info("Starting dataset retrieval")
        dataset = fetch_ucirepo(id=468)
        df: pd.DataFrame = dataset.data.features
        df_target: pd.DataFrame = dataset.data.targets
        df["revenue"] = df_target["Revenue"]
        logger.info(f"Dataset retrieval completed. Number of records: {len(df)}")

        logger.info("Starting date and user ID generation")
        df["SESSION_DATE"] = pd.to_datetime(
            df["Month"].apply(get_year)
            + df["Month"].map(MONTH_TO_NUM)
            + df["Month"].apply(get_random_day)
        )
        df["UID"] = [str(uuid.uuid4()) for _ in range(len(df))]
        logger.info("Date and user ID generation completed")

        logger.info(
            f"Starting data upload to Snowflake: {database_name}.{schema_name}.{table_name}"
        )
        upload_dataframe_to_snowflake(
            session=session,
            df=df,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
            mode=mode,
        )
        logger.info("Data upload to Snowflake completed")

    except Exception as e:
        logger.error(f"Error occurred during dataset preparation: {str(e)}")
        raise
