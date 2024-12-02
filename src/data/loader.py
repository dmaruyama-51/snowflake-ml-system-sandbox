import pandas as pd
from snowflake.snowpark import Session
import logging
from typing import Optional

SCHEMA = "ml"
TABLE = "online_shoppers_intention"
TARGET = "REVENUE"
CATEGORICAL_FEATURES = [
    "MONTH",
    "BROWSER",
    "REGION",
    "TRAFFICTYPE",
    "VISITORTYPE",
    "WEEKEND",
]
NUMERICAL_FEATURES = [
    "ADMINISTRATIVE",
    "ADMINISTRATIVE_DURATION",
    "INFORMATIONAL",
    "INFORMATIONAL_DURATION",
    "PRODUCTRELATED",
    "PRODUCTRELATED_DURATION",
    "BOUNCERATES",
    "EXITRATES",
    "PAGEVALUES",
    "SPECIALDAY",
]

logger = logging.getLogger(__name__)


def fetch_dataset(session: Session) -> Optional[pd.DataFrame]:
    """データセットを取得する関数

    Args:
        session (Session): Snowflakeセッション

    Returns:
        Optional[pd.DataFrame]: 取得したデータフレーム。エラー時はNone
    """

    try:
        # クエリ実行前のログ
        logger.info(f"{SCHEMA}.{TABLE}からデータセット取得を開始")

        select_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET]
        query_string = f"SELECT {', '.join(select_columns)} FROM {SCHEMA}.{TABLE}"
        df = session.sql(query_string).to_pandas()

        # 取得成功時のログ
        logger.info(f"データセット取得完了: {len(df)}行")
        return df

    except Exception as e:
        # エラー発生時のログ
        logger.error(f"データセット取得中にエラーが発生: {str(e)}")
        raise RuntimeError(f"データセット取得中にエラーが発生: {str(e)}")
