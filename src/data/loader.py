from typing import Optional
from snowflake.snowpark.session import Session
import pandas as pd
import logging
from snowflake.snowpark.exceptions import SnowparkSQLException

logger = logging.getLogger(__name__)


def load_data(
    session: Session,
    schema_name: str = "ml",
    table_name: str = "online_shoppers_intention",
) -> Optional[pd.DataFrame]:
    """
    Snowflakeからデータを読み込む

    Args:
        session: Snowflakeセッション
        table_name: 読み込むテーブル名

    Returns:
        pd.DataFrame: 読み込んだデータ
        None: エラー発生時
    """
    logger.info(f"データの読み込みを開始: {schema_name}.{table_name}")

    try:
        # セッションの状態確認
        if not session.is_active:
            raise ConnectionError("Snowflakeセッションが無効です")

        # クエリの実行
        query = f"SELECT * FROM {table_name}"
        df = session.sql(query).to_pandas()

        logger.info(f"データ読み込み完了: {len(df)} 行, {len(df.columns)} 列")
        return df

    except SnowparkSQLException as e:
        logger.error(f"SQLエラーが発生: {str(e)}")
        logger.error(f"SQLステート: {e.sqlstate}")
        raise

    except ConnectionError as e:
        logger.error(f"接続エラー: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"予期せぬエラーが発生: {str(e)}")
        raise

    finally:
        logger.debug("データ読み込み処理を終了")
