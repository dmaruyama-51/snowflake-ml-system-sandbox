import logging
from typing import Optional

import pandas as pd
from snowflake.snowpark import Session

from src.utils.config import load_config

logger = logging.getLogger(__name__)
config = load_config()


def fetch_dataset(session: Session, is_training: bool = True, prediction_date: str = None) -> Optional[pd.DataFrame]:
    """データセットを取得する関数

    Args:
        session (Session): Snowflakeセッション
        is_training (bool): 学習時かどうかのフラグ。デフォルトはTrue
        prediction_date (str): 推論日付（YYYY-MM-DD）

    Returns:
        Optional[pd.DataFrame]: 取得したデータフレーム。エラー時はNone
    """

    try:
        # config からデータセットの設定を取得
        schema = config["data"]["snowflake"]["schema"]
        table = config["data"]["snowflake"]["table"]
        categorical_features = config["data"]["features"]["categorical"]
        numerical_features = config["data"]["features"]["numeric"]
        target = config["data"]["target"]

        logger.info(f"{schema}.{table}からデータセット取得を開始")
        select_columns = ["UID"] + categorical_features + numerical_features + target

        # 学習時と推論時で日付条件を分岐
        if is_training:
            start_date = config["data"]["period"]["start_date"]
            end_date = config["data"]["period"]["end_date"]
            date_condition = f"SESSION_DATE BETWEEN '{start_date}' AND '{end_date}'"
            logger.info(f"学習用データを取得: 期間 {start_date} から {end_date}")
        else:

            if prediction_date is None:
                raise ValueError("推論時には prediction_date が必要です")
            
            # 推論時は引数の推論日付を条件にする
            date_condition = f"SESSION_DATE = '{prediction_date}'"
            logger.info(f"推論用データを取得: '{prediction_date}'")

        query_string = f"""
            SELECT {', '.join(select_columns)} 
            FROM {schema}.{table}
            WHERE {date_condition}
        """
        df = session.sql(query_string).to_pandas()

        if len(df) == 0:
            raise ValueError(f"指定期間のデータは存在しません。")

        # 取得成功時のログ
        logger.info(f"データセット取得完了: {len(df)}行")
        return df

    except Exception as e:
        # エラー発生時のログ
        logger.error(f"データセット取得中にエラーが発生: {str(e)}")
        raise RuntimeError(f"データセット取得中にエラーが発生: {str(e)}")
