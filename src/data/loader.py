from typing import Optional
from snowflake.snowpark.session import Session
import pandas as pd
import logging
from src.utils.snowflake import fetch_dataframe_from_snowflake
from src.utils.config import load_config

logger = logging.getLogger(__name__)
config = load_config()


def fetch_dataset(session: Session) -> Optional[pd.DataFrame]:
    """
    Snowflakeからデータを読み込む

    Args:
        session: Snowflakeセッション

    Returns:
        pd.DataFrame: 読み込んだデータ
        None: エラー発生時
    """
    schema = config["data"]["snowflake"]["schema"]
    table = config["data"]["snowflake"]["table"]
    logger.info(f"データの読み込みを開始: {schema}.{table}")

    target = config["data"]["target"]
    categorical_features = config["data"]["categorical_features"]
    numeric_features = config["data"]["numeric_features"]
    select_columns = categorical_features + numeric_features + [target]
    query = f"SELECT {', '.join(select_columns)} FROM {schema}.{table}"

    try:
        df = fetch_dataframe_from_snowflake(session, query)
        logger.info(f"データ読み込み完了: {len(df)} 行, {len(df.columns)} 列")
        return df

    except Exception as e:
        logger.error(f"予期せぬエラーが発生: {str(e)}")
        raise

    finally:
        logger.debug("データ読み込み処理を終了")
