import os
import json
import logging
from typing import Optional, List
import pandas as pd
from snowflake.snowpark import Session, DataFrame as SnowparkDataFrame
from snowflake.snowpark.exceptions import SnowparkSessionException, SnowparkSQLException

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


def fetch_dataframe_from_snowflake(session: Session, query: str) -> pd.DataFrame:
    """Snowflakeからデータを取得してDataFrameに変換"""
    if not query:
        raise ValueError("クエリが空です")

    try:
        logger.info(f"クエリ実行開始: {query[:100]}...")
        df = session.sql(query).to_pandas()
        logger.info(f"データ取得完了: {len(df)}行")
        return df

    except SnowparkSQLException as e:
        logger.error(f"SQLエラー: {e}")
        raise
    except Exception as e:
        logger.error(f"エラー発生: {e}")
        raise


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

def upload_files_to_stage(
    session: Session,
    local_path: str,
    stage_name: str,
    sub_path: str = None
) -> List[str]:
    """
    ローカルファイルをSnowflakeステージにアップロード

    Args:
        session: Snowflakeセッション
        local_path: アップロードするローカルディレクトリパス
        stage_name: ステージ名（例: "@practice.ml.modules"）
        sub_path: ステージ内のサブパス（例: "src"）

    Returns:
        List[str]: アップロードされたファイルのリスト
    """
    logger.info(f"ファイルアップロード開始: {stage_name}")
    logger.debug(f"パラメータ - local_path: {local_path}, stage_name: {stage_name}, sub_path: {sub_path}")

    try:
        uploaded_files = []
        
        # ディレクトリ内のファイルを再帰的に処理
        for root, _, files in os.walk(local_path):
            logger.debug(f"ディレクトリ処理中: {root}")
            for file in files:
                if file.endswith('.py') or file.endswith('.yml') or file.endswith('.json'):
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_path)
                    stage_path_sub = "/".join(relative_path.split("/")[:-1])
                    
                    # ステージパスの構築
                    stage_path = stage_name
                    if sub_path:
                        stage_path = f"{stage_name}/{sub_path}"
                    
                    logger.debug(f"アップロード試行 - ファイル: {relative_path}, ステージパス: {stage_path}")
                    
                    try:
                        # ファイルをアップロード
                        session.file.put(
                            local_file_path,
                            f"{stage_path}/{stage_path_sub}",
                            auto_compress=False, # True にすると圧縮ファイルとしてアップロードされる
                            overwrite=True
                        )
                        
                        uploaded_files.append(relative_path)
                        logger.info(f"アップロード成功: {relative_path}")
                    
                    except Exception as upload_error:
                        logger.error(f"個別ファイルのアップロード失敗 - {relative_path}: {str(upload_error)}")
                        raise
        
        logger.info(f"アップロード完了 - 合計ファイル数: {len(uploaded_files)}")
        return uploaded_files

    except Exception as e:
        logger.error(f"ファイルアップロード中にエラーが発生: {str(e)}", exc_info=True)
        raise RuntimeError(f"ファイルアップロード中にエラーが発生: {str(e)}")

    finally:
        logger.debug("アップロード処理終了")