import json
import logging
from typing import Optional

import pandas as pd
from snowflake.snowpark import (
    DataFrame as SnowparkDataFrame,
    Session,
)
from snowflake.snowpark.exceptions import SnowparkSessionException

logger = logging.getLogger(__name__)


def create_session() -> Optional[Session]:
    """
    snowpark session を作成

    Returns:
        Session: 成功時はSnowparkセッション、失敗時はNone

    Raises:
        SnowparkSessionException: Snowflakeへの接続に失敗した場合
    """
    try:
        logger.info("Starting Snowflake session creation")
        CONNECTION_PARAMETERS_PATH = "connection_parameters.json"
        logger.debug(f"Loading configuration file: {CONNECTION_PARAMETERS_PATH}")
        with open(CONNECTION_PARAMETERS_PATH) as f:
            connection_parameters = json.load(f)
        logger.info("Attempting to connect to Snowflake")
        session = Session.builder.configs(connection_parameters).create()
        logger.info("Successfully created Snowflake session")
        return session

    except SnowparkSessionException as e:
        error_msg = f"Failed to connect to Snowflake: {str(e)}"
        logger.error(error_msg)
        raise SnowparkSessionException(error_msg) from e


def upload_dataframe_to_snowflake(
    session: Session,
    df: pd.DataFrame,
    database_name: str,
    schema_name: str,
    table_name: str,
    mode: str = "overwrite",
) -> None:
    """
    Pandas DataFrameをSnowflakeにアップロードする

    Args:
        session (Session): Snowflakeセッション
        df (pd.DataFrame): アップロードするDataFrame
        database_name (str): ロード先のデータベース名
        schema_name (str): ロード先のスキーマ名
        table_name (str): ロード先のテーブル名
        mode (str, optional): データ書き込みモード. Defaults to 'overwrite'.
            'overwrite': テーブルを上書き
            'append': 既存テーブルにデータを追加
            'ignore': テーブルが存在する場合はスキップ
            'error': テーブルが存在する場合はエラー

    Raises:
        Exception: Snowflakeへのロード中にエラーが発生した場合
    """
    try:
        logger.info(
            f"Starting dataframe upload to: {database_name}.{schema_name}.{table_name}"
        )
        logger.info(f"Upload mode: {mode}")
        logger.info(f"Number of rows in dataframe: {len(df)}")
        logger.debug(f"Dataframe columns: {', '.join(df.columns)}")

        session.use_database(database_name)
        session.use_schema(schema_name)

        df.columns = df.columns.str.upper()

        full_table_name: str = f"{database_name}.{schema_name}.{table_name}"
        # appendモードでSESSION_DATEカラムが存在する場合、既存データを削除
        if mode == "append" and "SESSION_DATE" in df.columns:
            unique_dates = df["SESSION_DATE"].unique()

            logger.info(
                f"Deleting existing data: SESSION_DATE IN ({', '.join(map(str, unique_dates))})"
            )
            delete_sql = f"""
                DELETE FROM {full_table_name}
                WHERE SESSION_DATE IN ({','.join([f"'{date}'" for date in unique_dates])})
            """
            session.sql(delete_sql).collect()

        snowpark_df: SnowparkDataFrame = session.create_dataframe(df)
        logger.info(f"Starting write to table: {full_table_name}")
        snowpark_df.write.mode(mode).save_as_table(full_table_name)

        row_count: int = session.table(full_table_name).count()
        logger.info(f"Upload complete. Total rows in table: {row_count}")

    except Exception as e:
        error_msg = f"Failed to upload data: {str(e)}"
        logger.error(error_msg)
        raise
