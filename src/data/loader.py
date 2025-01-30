import logging
from typing import Optional

import pandas as pd
from snowflake.snowpark import Session

from src.utils.config import load_config

logger = logging.getLogger(__name__)
config = load_config()


def fetch_dataset(
    session: Session, is_training: bool = True, prediction_date: Optional[str] = None
) -> Optional[pd.DataFrame]:
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
        table = config["data"]["snowflake"]["dataset_table"]
        categorical_features = config["data"]["features"]["categorical"]
        numerical_features = config["data"]["features"]["numeric"]
        target = config["data"]["target"]

        logger.info(f"Starting dataset retrieval from {schema}.{table}")
        select_columns = ["UID"] + categorical_features + numerical_features + target

        # 学習時と推論時で日付条件を分岐
        if is_training:
            period_months = config["data"]["period"][
                "months"
            ]  # 期間（月数）を設定から取得
            # 現在の日付から指定された月数前までの期間を設定
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
            start_date = (
                pd.Timestamp.now() - pd.DateOffset(months=period_months)
            ).strftime("%Y-%m-%d")
            date_condition = f"SESSION_DATE BETWEEN '{start_date}' AND '{end_date}'"
            logger.info(
                f"Retrieving training data: period from {start_date} to {end_date} ({period_months} months)"
            )
        else:
            if prediction_date is None:
                raise ValueError("prediction_date is required for inference")

            # 推論時は引数の推論日付を条件にする
            date_condition = f"SESSION_DATE = '{prediction_date}'"
            logger.info(f"Retrieving inference data for date: '{prediction_date}'")

        query_string = f"""
            SELECT {', '.join(select_columns)} 
            FROM {schema}.{table}
            WHERE {date_condition}
        """
        df = session.sql(query_string).to_pandas()

        if len(df) == 0:
            raise ValueError("No data found for the specified period.")

        # 取得成功時のログ
        logger.info(f"Dataset retrieval completed: {len(df)} rows")
        return df

    except Exception as e:
        # エラー発生時のログ
        logger.error(f"Error occurred during dataset retrieval: {str(e)}")
        raise RuntimeError(f"Error occurred during dataset retrieval: {str(e)}")
