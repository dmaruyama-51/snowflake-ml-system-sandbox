import logging
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils.config import load_config

logger = logging.getLogger(__name__)
config = load_config()


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """データをテストデータとそれ以外とに分割"""
    test_size = 0.2
    random_state = 0

    logger.info(f"Starting data split (test_size: {test_size})")
    logger.debug(f"Input data shape: {df.shape}")

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Data split completed: train/val data {train_val.shape}, test data {test.shape}"
    )
    return train_val, test


def create_preprocessor() -> ColumnTransformer:
    """特徴量の前処理パイプラインを作成"""
    numeric_features = config["data"]["features"]["numeric"]
    categorical_features = config["data"]["features"]["categorical"]

    logger.info("Starting preprocessing pipeline creation")

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_features,
            ),
        ]
    )

    logger.info("Preprocessing pipeline creation completed")
    return preprocessor
