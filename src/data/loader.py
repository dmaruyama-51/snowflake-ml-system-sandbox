import logging
from datetime import datetime

import pandas as pd
from snowflake.ml.model import ModelVersion
from snowflake.snowpark import Session

from src.utils.config import load_config
from src.utils.constants import (
    CATEGORICAL_FEATURES,
    DATASET,
    NUMERICAL_FEATURES,
    SCHEMA,
    TARGET,
)

logger = logging.getLogger(__name__)
config = load_config()


def _get_base_config():
    """基本設定を取得する内部関数"""
    select_columns = ["UID"] + CATEGORICAL_FEATURES + NUMERICAL_FEATURES + TARGET
    return SCHEMA, DATASET, select_columns


def fetch_training_dataset(session: Session) -> pd.DataFrame:
    """学習用データセットを取得する関数

    Args:
        session (Session): Snowflakeセッション

    Returns:
        pd.DataFrame: 取得したデータフレーム
    """
    try:
        schema, table, select_columns = _get_base_config()
        period_months = config["data"]["period"]["months"]

        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        start_date = (
            pd.Timestamp.now() - pd.DateOffset(months=period_months)
        ).strftime("%Y-%m-%d")
        date_condition = f"SESSION_DATE BETWEEN '{start_date}' AND '{end_date}'"

        logger.info(
            f"Retrieving training data: period from {start_date} to {end_date} ({period_months} months)"
        )

        query_string = f"""
            SELECT {', '.join(select_columns)} 
            FROM {schema}.{table}
            WHERE {date_condition}
        """
        df = session.sql(query_string).to_pandas()

        if len(df) == 0:
            raise ValueError("No data found for the specified period.")

        logger.info(f"Training dataset retrieval completed: {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"Error occurred during training dataset retrieval: {str(e)}")
        raise RuntimeError(
            f"Error occurred during training dataset retrieval: {str(e)}"
        )


def fetch_prediction_dataset(session: Session, prediction_date: str) -> pd.DataFrame:
    """推論用データセットを取得する関数

    Args:
        session (Session): Snowflakeセッション
        prediction_date (str): 推論日付（YYYY-MM-DD）

    Returns:
        pd.DataFrame: 取得したデータフレーム
    """
    try:
        if not prediction_date:
            raise ValueError("prediction_date is required for inference")

        schema, table, select_columns = _get_base_config()
        date_condition = f"SESSION_DATE = '{prediction_date}'"

        logger.info(f"Retrieving inference data for date: {prediction_date}")

        query_string = f"""
            SELECT {', '.join(select_columns)} 
            FROM {schema}.{table}
            WHERE {date_condition}
        """
        df = session.sql(query_string).to_pandas()

        if len(df) == 0:
            raise ValueError("No data found for the specified date.")

        logger.info(f"Prediction dataset retrieval completed: {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"Error occurred during prediction dataset retrieval: {str(e)}")
        raise RuntimeError(
            f"Error occurred during prediction dataset retrieval: {str(e)}"
        )


def fetch_test_dataset(session: Session, model_version: ModelVersion) -> pd.DataFrame:
    """テスト用データセットを取得する関数

    Args:
        session (Session): Snowflakeセッション
        model_version (ModelVersion): 評価対象のモデルバージョン

    Returns:
        pd.DataFrame: 取得したデータフレーム
    """
    try:
        schema, table, select_columns = _get_base_config()

        # モデルバージョンの作成日を取得
        model_version_name = model_version.version_name
        model_created_date = datetime.strptime(f"20{model_version_name[2:8]}", "%Y%m%d")

        # 評価期間の設定（モデル作成日から2週間）
        start_date = (model_created_date + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        end_date = (model_created_date + pd.DateOffset(days=14)).strftime("%Y-%m-%d")
        date_condition = f"SESSION_DATE BETWEEN '{start_date}' AND '{end_date}'"

        logger.info(f"Retrieving testing data: period from {start_date} to {end_date}")

        query_string = f"""
            SELECT {', '.join(select_columns)} 
            FROM {schema}.{table}
            WHERE {date_condition}
        """
        df = session.sql(query_string).to_pandas()

        if len(df) == 0:
            raise ValueError("No data found for the specified period.")

        logger.info(f"Testing dataset retrieval completed: {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"Error occurred during testing dataset retrieval: {str(e)}")
        raise RuntimeError(f"Error occurred during testing dataset retrieval: {str(e)}")
