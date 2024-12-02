import pandas as pd
from snowflake.snowpark import Session
import logging
from typing import Optional
from src.utils.config import load_config

logger = logging.getLogger(__name__)
config = load_config()


def fetch_dataset(session: Session) -> Optional[pd.DataFrame]:
    """データセットを取得する関数

    Args:
        session (Session): Snowflakeセッション

    Returns:
        Optional[pd.DataFrame]: 取得したデータフレーム。エラー時はNone
    """

    try:
        schema = config["data"]["snowflake"]["schema"]
        table = config["data"]["snowflake"]["table"]
        categorical_features = config["data"]["features"]["categorical"]
        numerical_features = config["data"]["features"]["numeric"]
        target = config["data"]["target"]
        # クエリ実行前のログ
        logger.info(f"{schema}.{table}からデータセット取得を開始")

        select_columns = categorical_features + numerical_features + target
        query_string = f"SELECT {', '.join(select_columns)} FROM {schema}.{table}"
        df = session.sql(query_string).to_pandas()

        # 取得成功時のログ
        logger.info(f"データセット取得完了: {len(df)}行")
        return df

    except Exception as e:
        # エラー発生時のログ
        logger.error(f"データセット取得中にエラーが発生: {str(e)}")
        raise RuntimeError(f"データセット取得中にエラーが発生: {str(e)}")
