import json
import logging
from typing import Optional
import pandas as pd
from snowflake.snowpark import Session, DataFrame as SnowparkDataFrame
from snowflake.snowpark.exceptions import SnowparkSessionException

logger = logging.getLogger(__name__)
CONNECTION_PARAMETERS_PATH = "connection_parameters.json"


def create_session() -> Optional[Session]:
    """
    snowpark session を作成

    Returns:
        Session: 成功時はSnowparkセッション、失敗時はNone

    Raises:
        SnowparkSessionException: Snowflakeへの接続に失敗した場合
    """
    try:
        logger.info("Snowflakeセッションの作成を開始")
        logger.debug(f"設定ファイルを読み込み: {CONNECTION_PARAMETERS_PATH}")
        with open(CONNECTION_PARAMETERS_PATH) as f:
            connection_parameters = json.load(f)
        logger.info("Snowflakeへの接続を試行")
        session = Session.builder.configs(connection_parameters).create()
        logger.info("Snowflakeセッションの作成に成功")
        return session

    except SnowparkSessionException as e:
        error_msg = f"Snowflakeへの接続に失敗しました: {str(e)}"
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
            f"データフレームのアップロードを開始: {database_name}.{schema_name}.{table_name}"
        )
        logger.info(f"アップロードモード: {mode}")
        logger.info(f"データフレームの行数: {len(df)}")
        logger.debug(f"データフレームのカラム: {', '.join(df.columns)}")

        session.use_database(database_name)
        session.use_schema(schema_name)

        df.columns = df.columns.str.upper()
        snowpark_df: SnowparkDataFrame = session.create_dataframe(df)
        full_table_name: str = f"{database_name}.{schema_name}.{table_name}"
        logger.info(f"テーブルへの書き込みを開始: {full_table_name}")
        snowpark_df.write.mode(mode).save_as_table(full_table_name)

        row_count: int = session.table(full_table_name).count()
        logger.info(f"アップロード完了。テーブルの総行数: {row_count}")

    except Exception as e:
        error_msg = f"データのアップロードに失敗しました: {str(e)}"
        logger.error(error_msg)
        raise
